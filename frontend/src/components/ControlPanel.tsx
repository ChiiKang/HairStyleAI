interface Props {
  intensity: number;
  lift: number;
  onIntensityChange: (v: number) => void;
  onLiftChange: (v: number) => void;
}

export function ControlPanel({
  intensity,
  lift,
  onIntensityChange,
  onLiftChange,
}: Props) {
  return (
    <div className="space-y-4">
      <div>
        <label className="mb-1 flex justify-between text-sm font-medium text-gray-300">
          <span>Color Intensity</span>
          <span className="text-gray-500">{intensity}%</span>
        </label>
        <input
          type="range"
          min={0}
          max={100}
          value={intensity}
          onChange={(e) => onIntensityChange(Number(e.target.value))}
          className="w-full"
        />
      </div>
      <div>
        <label className="mb-1 flex justify-between text-sm font-medium text-gray-300">
          <span>Dark Hair Lift</span>
          <span className="text-gray-500">{lift}%</span>
        </label>
        <input
          type="range"
          min={0}
          max={40}
          value={lift}
          onChange={(e) => onLiftChange(Number(e.target.value))}
          className="w-full"
        />
        <p className="mt-1 text-xs text-gray-500">
          Brightens dark hair so color shows better
        </p>
      </div>
    </div>
  );
}
