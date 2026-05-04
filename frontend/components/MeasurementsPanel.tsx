"use client";

import { parseMeasurements } from "@/lib/api";

type Props = {
  text: string | null;
};

const ICONS: Record<string, string> = {
  "Chest":          "⬤",
  "Waist":          "⬤",
  "Hip":            "⬤",
  "Shoulder width": "⬤",
  "Inseam":         "⬤",
  "Arm length":     "⬤",
};

export default function MeasurementsPanel({ text }: Props) {
  if (!text) {
    return (
      <div className="flex flex-col gap-2">
        <h2 className="text-sm font-medium text-gray-500 uppercase tracking-wider">Measurements</h2>
        <div className="rounded-xl border border-gray-200 p-4">
          <p className="text-sm text-gray-400 text-center py-4">
            Upload a photo to see your measurements
          </p>
        </div>
      </div>
    );
  }

  const parsed = parseMeasurements(text);
  const entries = Object.entries(parsed);

  return (
    <div className="flex flex-col gap-2">
      <h2 className="text-sm font-medium text-gray-500 uppercase tracking-wider">Measurements</h2>
      <div className="rounded-xl border border-gray-200 divide-y divide-zinc-800">
        {entries.length > 0 ? (
          entries.map(([label, value]) => (
            <div key={label} className="flex justify-between items-center px-4 py-3">
              <span className="text-sm text-gray-700">{label}</span>
              <span className="text-sm font-medium tabular-nums text-gray-900 font-semibold">{value}</span>
            </div>
          ))
        ) : (
          <div className="px-4 py-3">
            <pre className="text-xs text-gray-500 whitespace-pre-wrap font-mono">{text}</pre>
          </div>
        )}
      </div>
      <p className="text-xs text-gray-400">
        Estimates from a single photo. Accuracy improves with a fitted, front-facing photo.
      </p>
    </div>
  );
}
