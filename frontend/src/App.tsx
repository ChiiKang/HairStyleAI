import { useState, useCallback } from "react";
import { ImageUploader } from "./components/ImageUploader";
import { ModelSelector } from "./components/ModelSelector";
import { ColorPicker } from "./components/ColorPicker";
import { ControlPanel } from "./components/ControlPanel";
import { ResultViewer } from "./components/ResultViewer";
import { useHairRecolor } from "./hooks/useHairRecolor";
import type { SegmentationModel } from "./types";

function App() {
  const [imageFile, setImageFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [model, setModel] = useState<SegmentationModel>("bisenet");
  const [color, setColor] = useState("#D4A574");
  const [intensity, setIntensity] = useState(80);
  const [lift, setLift] = useState(0);

  const { maskUrl, resultUrl, loading, error, processImage, reset } =
    useHairRecolor();

  const handleImageSelected = useCallback((file: File, url: string) => {
    setImageFile(file);
    setPreviewUrl(url);
  }, []);

  const handleProcess = useCallback(() => {
    if (!imageFile) return;
    processImage(imageFile, model, color, intensity, lift);
  }, [imageFile, model, color, intensity, lift, processImage]);

  const handleReset = useCallback(() => {
    setImageFile(null);
    setPreviewUrl(null);
    reset();
  }, [reset]);

  return (
    <div className="min-h-screen bg-gray-950 text-white">
      <header className="border-b border-gray-800 px-6 py-4">
        <h1 className="text-2xl font-bold">Hair Color Change</h1>
        <p className="text-sm text-gray-400">
          Upload a photo and try different hair colors
        </p>
      </header>

      <main className="mx-auto max-w-6xl p-6">
        <div className="grid gap-6 lg:grid-cols-[320px_1fr]">
          {/* Sidebar controls */}
          <div className="space-y-6">
            <ModelSelector value={model} onChange={setModel} />
            <ColorPicker value={color} onChange={setColor} />
            <ControlPanel
              intensity={intensity}
              lift={lift}
              onIntensityChange={setIntensity}
              onLiftChange={setLift}
            />

            <button
              onClick={handleProcess}
              disabled={!imageFile || loading}
              className="w-full rounded-lg bg-indigo-600 px-4 py-3 font-medium text-white transition hover:bg-indigo-500 disabled:opacity-40 disabled:cursor-not-allowed"
            >
              {loading ? "Processing..." : "Apply Color"}
            </button>

            {resultUrl && (
              <button
                onClick={handleReset}
                className="w-full rounded-lg border border-gray-700 px-4 py-2 text-sm text-gray-300 transition hover:bg-gray-800"
              >
                Start Over
              </button>
            )}

            {error && (
              <p className="rounded-lg bg-red-900/30 px-3 py-2 text-sm text-red-400">
                {error}
              </p>
            )}
          </div>

          {/* Main content area */}
          <div className="space-y-6">
            {!resultUrl && (
              <ImageUploader
                onImageSelected={handleImageSelected}
                previewUrl={previewUrl}
              />
            )}
            <ResultViewer
              originalUrl={previewUrl}
              resultUrl={resultUrl}
              maskUrl={maskUrl}
            />
          </div>
        </div>
      </main>
    </div>
  );
}

export default App;
