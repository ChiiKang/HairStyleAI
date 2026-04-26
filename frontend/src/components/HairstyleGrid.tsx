import { HairstyleCard } from "./HairstyleCard";

interface GeneratedResult {
  imageUrl: string | null;
  label: string;
}

interface Props {
  results: GeneratedResult[];
  loading: boolean;
  model: string;
  durationMs: number | null;
}

const DEFAULT_LABELS = [
  "Protective Braids",
  "Natural Twist-Out",
  "Silk Press",
  "Bantu Knots",
];

export function HairstyleGrid({ results, loading, model, durationMs }: Props) {
  const cards = results.length > 0
    ? results
    : DEFAULT_LABELS.map((label) => ({ imageUrl: null, label }));

  return (
    <div className="space-y-3">
      <div className="flex items-center justify-between">
        <h3 className="text-sm font-medium text-gray-300">Generated Hairstyles</h3>
        {durationMs !== null && !loading && (
          <span className="text-xs text-gray-500">
            {model.split("/").pop()} — {(durationMs / 1000).toFixed(1)}s
          </span>
        )}
      </div>

      <div className="grid grid-cols-2 gap-3">
        {cards.map((card, i) => (
          <HairstyleCard
            key={i}
            imageUrl={card.imageUrl}
            label={card.label}
            loading={loading}
            index={i}
          />
        ))}
      </div>
    </div>
  );
}
