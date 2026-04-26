interface Props {
  value: string;
  onChange: (model: string) => void;
}

const MODELS = [
  {
    id: "openai/gpt-image-2",
    label: "GPT Image 2 Edit",
    desc: "OpenAI, high quality editing",
    cost: "$0.04-0.17/img",
  },
  {
    id: "fal-ai/qwen-image-edit-plus",
    label: "Qwen Image Edit Plus",
    desc: "Multi-image editing",
    cost: "$0.03/MP",
  },
  {
    id: "fal-ai/luma-photon/flash",
    label: "Luma Photon Modify",
    desc: "Fast image editing",
    cost: "$0.005/MP",
  },
  {
    id: "fal-ai/bytedance/seedream/v5/lite/text-to-image",
    label: "Seedream 5.0 Edit",
    desc: "ByteDance image editing",
    cost: "$0.035/img",
  },
];

export function GenerationModelSelector({ value, onChange }: Props) {
  return (
    <div>
      <label className="mb-1 block text-sm font-medium text-gray-300">
        Generation Model
      </label>
      <select
        value={value}
        onChange={(e) => onChange(e.target.value)}
        className="w-full rounded-lg border border-gray-700 bg-gray-800 px-3 py-2 text-white"
      >
        {MODELS.map((m) => (
          <option key={m.id} value={m.id}>
            {m.label} — {m.desc} ({m.cost})
          </option>
        ))}
      </select>
    </div>
  );
}
