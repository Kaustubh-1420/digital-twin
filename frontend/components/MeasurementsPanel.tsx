"use client";

import { parseMeasurements } from "@/lib/api";

type Props = {
  text: string | null;
};

const LABELS: { key: string; label: string }[] = [
  { key: "Chest",          label: "Chest" },
  { key: "Waist",          label: "Waist" },
  { key: "Hip",            label: "Hip" },
  { key: "Shoulder width", label: "Shoulder width" },
  { key: "Inseam",         label: "Inseam" },
  { key: "Arm length",     label: "Arm length" },
];

export default function MeasurementsPanel({ text }: Props) {
  const parsed = text ? parseMeasurements(text) : {};
  const hasData = Object.keys(parsed).length > 0;

  return (
    <div className="flex flex-col gap-3">
      <h2 className="text-[11px] font-medium uppercase tracking-wider text-black/50">
        Measurements
      </h2>

      <div className="rounded-xl border border-black/[0.07] overflow-hidden bg-white">
        {LABELS.map(({ key, label }, i) => {
          const value = (parsed as Record<string, string>)[key];
          const isLast = i === LABELS.length - 1;
          return (
            <div
              key={key}
              className={`flex justify-between items-center px-4 py-3 ${isLast ? "" : "border-b border-black/[0.06]"}`}
            >
              <span className="text-[12.5px] text-black/65">{label}</span>
              <span
                className={`text-[13px] font-medium tabular-nums ${value ? "text-[#0c0c0a]" : "text-black/25"}`}
              >
                {value ?? "— cm"}
              </span>
            </div>
          );
        })}
      </div>

      <p className="text-[11px] text-black/45 leading-relaxed px-1">
        {hasData
          ? "Estimates from a single photo. Accuracy improves with a fitted, front-facing photo."
          : "Upload a photo to generate measurements."}
      </p>
    </div>
  );
}
