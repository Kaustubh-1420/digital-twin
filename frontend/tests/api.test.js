/**
 * Tests for lib/api.ts — the pipeline client.
 *
 * These cover the failure modes behind the 503 bug: /api/glb returning a
 * transient 503 on the payload path, and /api/predict returning a platform
 * HTML error page instead of JSON. Both produced a page that looked like a
 * success while showing the stock mannequin.
 *
 * Run with `npm test`, which compiles lib/api.ts to .test-build first.
 * lib/api.ts has no React or Next imports, so it runs directly under Node.
 */

const { runPipeline, GlbDownloadError } = require("../.test-build/api.js");

// --- browser shims -------------------------------------------------------

globalThis.URL.createObjectURL = (blob) => `blob:test-${blob.size}`;
globalThis.URL.revokeObjectURL = () => {};

const GLB_BYTES = new Uint8Array(1024).fill(7);
const OK_PREDICT = {
  glbUrl: "https://kaustubh1420-digital-twin.hf.space/gradio_api/file=/tmp/x.glb",
  measurements: "Chest              88.1 cm",
};

let predictCalls = 0;

/** Stub fetch: `predict` builds the /api/predict response, `glbStatuses` is
 *  consumed one entry per /api/glb attempt (last entry repeats). */
function stubFetch({ predict, glbStatuses = [200] }) {
  const glbCalls = [];
  let i = 0;
  predictCalls = 0;
  globalThis.fetch = async (url) => {
    const u = String(url);
    if (u.includes("/api/predict")) {
      const n = predictCalls++;
      return predict
        ? predict(n)
        : new Response(JSON.stringify(OK_PREDICT), { status: 200 });
    }
    glbCalls.push(u);
    const status = glbStatuses[Math.min(i++, glbStatuses.length - 1)];
    return status === 200
      ? new Response(GLB_BYTES, { status: 200 })
      : new Response(`Upstream error ${status}`, { status });
  };
  return glbCalls;
}

// --- runner --------------------------------------------------------------

const results = [];

async function test(name, setup, expect) {
  const glbCalls = stubFetch(setup);
  const started = Date.now();
  let outcome;
  try {
    const r = await runPipeline(new Blob([new Uint8Array([1, 2, 3])]), 170);
    outcome = { ok: true, glbUrl: r.glbUrl, measurements: r.measurements };
  } catch (e) {
    outcome = {
      ok: false,
      message: e.message,
      glbOnly: e instanceof GlbDownloadError,
      measurements: e instanceof GlbDownloadError ? e.measurements : undefined,
    };
  }

  const attempts = glbCalls.length;
  let pass = false;
  let detail = "";
  try {
    pass = expect(outcome, attempts, predictCalls);
  } catch (e) {
    detail = ` (assertion threw: ${e.message})`;
  }

  console.log(`\n  ${pass ? "PASS" : "FAIL"}  ${name}`);
  console.log(`        predict calls: ${predictCalls}, glb attempts: ${attempts}, elapsed: ${Date.now() - started}ms`);
  console.log(
    `        ${outcome.ok ? `resolved ${outcome.glbUrl}` : `rejected "${outcome.message}"`}${detail}`
  );
  results.push(pass);
}

// --- cases ---------------------------------------------------------------

(async () => {
  console.log("\nGLB delivery — retry on transient failure");

  // The exact sequence captured in the browser against production.
  await test(
    "503 twice then 200 recovers without surfacing an error",
    { glbStatuses: [503, 503, 200] },
    (o, n) => o.ok && n === 3
  );

  await test(
    "503 on every attempt fails honestly after 3 tries",
    { glbStatuses: [503] },
    (o, n) => !o.ok && n === 3 && /could not be downloaded/.test(o.message)
  );

  // Estimation succeeded, so its measurements belong to this photo and must
  // survive a failed 3D download instead of being thrown away with it.
  await test(
    "GLB failure keeps the measurements from the successful estimation",
    { glbStatuses: [503] },
    (o) => !o.ok && o.glbOnly && o.measurements === OK_PREDICT.measurements
  );

  await test(
    "200 on first try does not retry",
    { glbStatuses: [200] },
    (o, n) => o.ok && n === 1
  );

  await test(
    "403 is permanent and fails fast without backoff",
    { glbStatuses: [403] },
    (o, n) => !o.ok && n === 1 && o.glbOnly
  );

  console.log("\nPredict response — parsed defensively");

  // A cold function invocation returns Vercel's HTML error page. Calling
  // res.json() on that threw before res.ok was ever checked, surfacing to
  // the user as "Unexpected token 'A'".
  await test(
    "HTML error page reports the status, not a JSON parse error",
    {
      predict: () =>
        new Response("An error occurred with this application.", {
          status: 500,
          headers: { "content-type": "text/html" },
        }),
    },
    (o) => !o.ok && /Body estimation failed \(HTTP 500\)/.test(o.message)
  );

  // No estimation happened, so there is nothing to keep.
  await test(
    "predict failure carries no measurements",
    { predict: () => new Response("Gateway Timeout", { status: 504 }) },
    (o) => !o.ok && !o.glbOnly && o.measurements === undefined
  );

  await test(
    "JSON error body surfaces the server's own message",
    {
      predict: () =>
        new Response(JSON.stringify({ error: "No person detected in photo." }), {
          status: 502,
        }),
    },
    (o) => !o.ok && o.message === "No person detected in photo."
  );

  await test(
    "200 with a non-JSON body is treated as a failure",
    { predict: () => new Response("not json at all", { status: 200 }) },
    (o) => !o.ok && /unexpected response/.test(o.message)
  );

  await test(
    "200 missing glbUrl is treated as a failure",
    {
      predict: () =>
        new Response(JSON.stringify({ measurements: "Chest  88.1 cm" }), {
          status: 200,
        }),
    },
    (o) => !o.ok && /unexpected response/.test(o.message)
  );

  console.log("\nPredict retry — cold or restarting backend");

  const ok = () => new Response(JSON.stringify(OK_PREDICT), { status: 200 });

  await test(
    "504 then 200 recovers on the single retry",
    { predict: (n) => (n === 0 ? new Response("Gateway Timeout", { status: 504 }) : ok()) },
    (o, _g, p) => o.ok && p === 2
  );

  await test(
    "infrastructure 502 (retryable) then 200 recovers",
    {
      predict: (n) =>
        n === 0
          ? new Response(JSON.stringify({ error: "Connection errored out.", retryable: true }), { status: 502 })
          : ok(),
    },
    (o, _g, p) => o.ok && p === 2
  );

  // "No person detected" is the app's answer for this photo; retrying only
  // makes the user wait for the same answer.
  await test(
    "app-level 502 is not retried",
    {
      predict: () =>
        new Response(JSON.stringify({ error: "No person detected in photo." }), { status: 502 }),
    },
    (o, _g, p) => !o.ok && p === 1 && o.message === "No person detected in photo."
  );

  await test(
    "504 on both attempts gives up after one retry",
    { predict: () => new Response("Gateway Timeout", { status: 504 }) },
    (o, _g, p) => !o.ok && p === 2 && /HTTP 504/.test(o.message)
  );

  await test(
    "healthy response resolves with measurements intact",
    {},
    (o) => o.ok && o.measurements === OK_PREDICT.measurements
  );

  const passed = results.filter(Boolean).length;
  console.log(`\n${passed}/${results.length} passed\n`);
  process.exit(passed === results.length ? 0 : 1);
})();
