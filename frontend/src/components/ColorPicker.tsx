interface Props {
  value: string;
  onChange: (color: string) => void;
}

const PRESETS = [
  { color: "#D4A574", label: "Honey Blonde" },
  { color: "#8B4513", label: "Auburn" },
  { color: "#FF4444", label: "Red" },
  { color: "#FF69B4", label: "Pink" },
  { color: "#9B59B6", label: "Purple" },
  { color: "#3498DB", label: "Blue" },
  { color: "#2ECC71", label: "Green" },
  { color: "#F1C40F", label: "Gold" },
  { color: "#E0E0E0", label: "Platinum" },
  { color: "#1A1A1A", label: "Black" },
];

export function ColorPicker({ value, onChange }: Props) {
  return (
    <div>
      <label className="mb-1 block text-sm font-medium text-gray-300">
        Hair Color
      </label>
      <div className="flex flex-wrap gap-2">
        {PRESETS.map((p) => (
          <button
            key={p.color}
            title={p.label}
            onClick={() => onChange(p.color)}
            className={`h-8 w-8 rounded-full border-2 transition ${
              value === p.color ? "border-white scale-110" : "border-gray-600"
            }`}
            style={{ backgroundColor: p.color }}
          />
        ))}
      </div>
      <div className="mt-2 flex items-center gap-2">
        <input
          type="color"
          value={value}
          onChange={(e) => onChange(e.target.value)}
          className="h-8 w-8 cursor-pointer rounded border-0 bg-transparent"
        />
        <input
          type="text"
          value={value}
          onChange={(e) => onChange(e.target.value)}
          className="w-24 rounded border border-gray-700 bg-gray-800 px-2 py-1 text-sm text-white"
        />
      </div>
    </div>
  );
}
