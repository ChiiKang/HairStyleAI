interface Props {
  value: string;
  onChange: (model: string) => void;
}

const MODELS = [
  // Recommended
  {
    id: "fal-ai/flux-kontext/dev",
    label: "FLUX Kontext",
    desc: "Best quality, face-safe",
    cost: "$0.025/MP",
  },
  {
    id: "fal-ai/flux-2/edit",
    label: "FLUX.2 Edit",
    desc: "Next-gen editing",
    cost: "$0.024/img",
  },
  {
    id: "xai/grok-imagine-image/edit",
    label: "Grok Imagine Edit",
    desc: "Cheapest flat-rate",
    cost: "$0.022/img",
  },
  {
    id: "fal-ai/chrono-edit-lora",
    label: "Chrono Edit LoRA",
    desc: "NVIDIA physics-aware",
    cost: "$0.02/img",
  },
  {
    id: "fal-ai/image-editing/hair-change",
    label: "Hair Change (Dedicated)",
    desc: "Purpose-built for hair",
    cost: "$0.04/img",
  },
  // Original models
  {
    id: "openai/gpt-image-2",
    label: "GPT Image 2 Edit",
    desc: "OpenAI, high quality",
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
    desc: "ByteDance editing",
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
