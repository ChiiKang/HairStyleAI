import { useCallback, useRef } from "react";

interface Props {
  onImageSelected: (file: File, previewUrl: string) => void;
  previewUrl: string | null;
}

export function ImageUploader({ onImageSelected, previewUrl }: Props) {
  const inputRef = useRef<HTMLInputElement>(null);

  const handleFile = useCallback(
    (file: File) => {
      const url = URL.createObjectURL(file);
      onImageSelected(file, url);
    },
    [onImageSelected]
  );

  const handleDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault();
      const file = e.dataTransfer.files[0];
      if (file && file.type.startsWith("image/")) handleFile(file);
    },
    [handleFile]
  );

  return (
    <div
      onDrop={handleDrop}
      onDragOver={(e) => e.preventDefault()}
      onClick={() => inputRef.current?.click()}
      className="cursor-pointer rounded-xl border-2 border-dashed border-gray-700 p-8 text-center transition hover:border-gray-500"
    >
      {previewUrl ? (
        <img
          src={previewUrl}
          alt="Uploaded"
          className="mx-auto max-h-80 rounded-lg object-contain"
        />
      ) : (
        <div className="space-y-2 text-gray-400">
          <p className="text-lg">Drop an image here or click to upload</p>
          <p className="text-sm">JPG, PNG supported</p>
        </div>
      )}
      <input
        ref={inputRef}
        type="file"
        accept="image/*"
        className="hidden"
        onChange={(e) => {
          const file = e.target.files?.[0];
          if (file) handleFile(file);
        }}
      />
    </div>
  );
}
