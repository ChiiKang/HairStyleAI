import { useState, useCallback } from "react";
import { SelfieCapture } from "./SelfieCapture";
import { GenerationModelSelector } from "./GenerationModelSelector";
import { HairstyleGrid } from "./HairstyleGrid";
import { Gallery } from "./Gallery";
import { useHairstyleGenerator } from "../hooks/useHairstyleGenerator";

export function HairstyleGenerator() {
  const [imageFile, setImageFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [model, setModel] = useState("fal-ai/flux-kontext/dev");
  const [galleryKey, setGalleryKey] = useState(0);

  const { results, loading, error, model: usedModel, durationMs, generate, reset } =
    useHairstyleGenerator();

  const handleImageCaptured = useCallback((file: File, url: string) => {
    if (!file) {
      setImageFile(null);
      setPreviewUrl(null);
      reset();
      return;
    }
    setImageFile(file);
    setPreviewUrl(url);
    reset();
  }, [reset]);

  const handleGenerate = useCallback(async () => {
    if (!imageFile) return;
    await generate(imageFile, model);
    // Refresh gallery after generation completes
    setGalleryKey((k) => k + 1);
  }, [imageFile, model, generate]);

  const handleStartOver = useCallback(() => {
    setImageFile(null);
    setPreviewUrl(null);
    reset();
  }, [reset]);

  return (
    <div className="grid gap-6 lg:grid-cols-[320px_1fr]">
      {/* Sidebar */}
      <div className="space-y-6">
        <SelfieCapture
          onImageCaptured={handleImageCaptured}
          previewUrl={previewUrl}
        />

        <GenerationModelSelector value={model} onChange={setModel} />

        <button
          onClick={handleGenerate}
          disabled={!imageFile || loading}
          className="w-full rounded-lg bg-indigo-600 px-4 py-3 font-medium text-white transition hover:bg-indigo-500 disabled:opacity-40 disabled:cursor-not-allowed"
        >
          {loading ? "Generating..." : "Generate Hairstyles"}
        </button>

        {results.length > 0 && (
          <button
            onClick={handleStartOver}
            className="w-full rounded-lg border border-gray-700 px-4 py-2 text-sm text-gray-300 transition hover:bg-gray-800"
          >
            Start Over
          </button>
        )}

        {error && (
          <div className="rounded-lg bg-red-900/30 px-3 py-2 text-sm text-red-400">
            {error}
            {error.includes("Generation failed") && (
              <p className="mt-1 text-xs text-red-500/70">
                Backend endpoint /api/generate-hairstyles may not be running yet.
              </p>
            )}
          </div>
        )}

        <div className="rounded-lg bg-gray-900 px-3 py-2 text-xs text-gray-500 space-y-1">
          <p className="font-medium text-gray-400">How it works</p>
          <p>Upload your photo or take a selfie. The AI generates 4 hairstyle variations tailored to your face shape and features.</p>
          <p>All generated images are saved locally and viewable below.</p>
        </div>
      </div>

      {/* Main content */}
      <div className="space-y-8">
        <HairstyleGrid
          results={results}
          loading={loading}
          model={usedModel}
          durationMs={durationMs}
        />

        <Gallery key={galleryKey} />
      </div>
    </div>
  );
}
