interface Props {
  imageUrl: string | null;
  label: string;
  loading: boolean;
  index: number;
}

export function HairstyleCard({ imageUrl, label, loading, index }: Props) {
  return (
    <div className="overflow-hidden rounded-xl border border-gray-800 bg-gray-900">
      <div className="aspect-square relative">
        {loading ? (
          <div className="absolute inset-0 flex flex-col items-center justify-center gap-3 bg-gray-900">
            <div className="h-8 w-8 animate-spin rounded-full border-2 border-gray-600 border-t-indigo-400" />
            <p className="text-xs text-gray-500">Generating style {index + 1}...</p>
          </div>
        ) : imageUrl ? (
          <img
            src={imageUrl}
            alt={label}
            className="h-full w-full object-cover"
          />
        ) : (
          <div className="absolute inset-0 flex items-center justify-center bg-gray-900">
            <p className="text-sm text-gray-600">Style {index + 1}</p>
          </div>
        )}
      </div>
      <div className="flex items-center justify-between px-3 py-2">
        <p className="text-xs font-medium text-gray-300 truncate">{label}</p>
        {imageUrl && (
          <a
            href={imageUrl}
            download={`hairstyle-${index + 1}.png`}
            className="text-xs text-indigo-400 hover:text-indigo-300 transition"
          >
            Save
          </a>
        )}
      </div>
    </div>
  );
}
