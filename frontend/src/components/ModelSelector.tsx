import type { SegmentationModel } from "../types";

interface Props {
  value: SegmentationModel;
  onChange: (model: SegmentationModel) => void;
}

const MODELS: { value: SegmentationModel; label: string; desc: string }[] = [
  { value: "bisenet", label: "BiSeNet", desc: "Proven classic, fast" },
  { value: "fashn", label: "FASHN", desc: "Fashion-optimized, full body" },
  { value: "segformer", label: "SegFormer", desc: "High quality face parsing" },
];

export function ModelSelector({ value, onChange }: Props) {
  return (
    <div>
      <label className="mb-1 block text-sm font-medium text-gray-300">
        Segmentation Model
      </label>
      <select
        value={value}
        onChange={(e) => onChange(e.target.value as SegmentationModel)}
        className="w-full rounded-lg border border-gray-700 bg-gray-800 px-3 py-2 text-white"
      >
        {MODELS.map((m) => (
          <option key={m.value} value={m.value}>
            {m.label} — {m.desc}
          </option>
        ))}
      </select>
    </div>
  );
}
