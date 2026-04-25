import { useState } from "react";

interface Props {
  originalUrl: string | null;
  resultUrl: string | null;
  maskUrl: string | null;
}

type ViewMode = "result" | "before-after" | "mask";

export function ResultViewer({ originalUrl, resultUrl, maskUrl }: Props) {
  const [mode, setMode] = useState<ViewMode>("result");

  if (!resultUrl || !originalUrl) return null;

  return (
    <div className="space-y-3">
      <div className="flex gap-2">
        {(["result", "before-after", "mask"] as ViewMode[]).map((m) => (
          <button
            key={m}
            onClick={() => setMode(m)}
            className={`rounded-lg px-3 py-1 text-sm ${
              mode === m
                ? "bg-white text-black"
                : "bg-gray-800 text-gray-300 hover:bg-gray-700"
            }`}
          >
            {m === "before-after" ? "Before / After" : m.charAt(0).toUpperCase() + m.slice(1)}
          </button>
        ))}
      </div>

      <div className="overflow-hidden rounded-xl border border-gray-800">
        {mode === "result" && (
          <img src={resultUrl} alt="Result" className="w-full object-contain" />
        )}
        {mode === "before-after" && (
          <div className="grid grid-cols-2 gap-1">
            <img src={originalUrl} alt="Before" className="w-full object-contain" />
            <img src={resultUrl} alt="After" className="w-full object-contain" />
          </div>
        )}
        {mode === "mask" && maskUrl && (
          <img src={maskUrl} alt="Hair mask" className="w-full object-contain" />
        )}
      </div>

      <a
        href={resultUrl}
        download="hair-recolored.png"
        className="inline-block rounded-lg bg-white px-4 py-2 text-sm font-medium text-black transition hover:bg-gray-200"
      >
        Download Result
      </a>
    </div>
  );
}
