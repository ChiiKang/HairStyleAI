import { useState, useCallback, useEffect, useRef } from "react";
import { TabBar } from "./components/TabBar";
import { ImageUploader } from "./components/ImageUploader";
import { ModelSelector } from "./components/ModelSelector";
import { ColorPicker } from "./components/ColorPicker";
import { ControlPanel } from "./components/ControlPanel";
import { ResultViewer } from "./components/ResultViewer";
import { HairstyleGenerator } from "./components/HairstyleGenerator";
import { useHairRecolor } from "./hooks/useHairRecolor";
import type { SegmentationModel, RecolorMethod } from "./types";

function ColorChangeTab() {
  const [imageFile, setImageFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [model, setModel] = useState<SegmentationModel>("bisenet");
  const [color, setColor] = useState("#D4A574");
  const [intensity, setIntensity] = useState(80);
  const [lift, setLift] = useState(0);
  const [method, setMethod] = useState<RecolorMethod>("reinhard");

  const { maskUrl, resultUrl, segmenting, hasMask, error, segment, recolor, reset } =
    useHairRecolor();

  const handleImageSelected = useCallback((file: File, url: string) => {
    setImageFile(file);
    setPreviewUrl(url);
  }, []);

  const handleProcess = useCallback(async () => {
    if (!imageFile) return;
    const ok = await segment(imageFile, model);
    if (ok) {
      recolor(color, intensity, lift, method);
    }
  }, [imageFile, model, color, intensity, lift, method, segment, recolor]);

  const debounceRef = useRef<ReturnType<typeof setTimeout>>();
  useEffect(() => {
    if (!hasMask) return;
    clearTimeout(debounceRef.current);
    debounceRef.current = setTimeout(() => {
      recolor(color, intensity, lift, method);
    }, 150);
    return () => clearTimeout(debounceRef.current);
  }, [color, intensity, lift, method, hasMask, recolor]);

  const handleReset = useCallback(() => {
    setImageFile(null);
    setPreviewUrl(null);
    reset();
  }, [reset]);

  return (
    <div className="grid gap-6 lg:grid-cols-[320px_1fr]">
      <div className="space-y-6">
        <ModelSelector value={model} onChange={setModel} />

        <div>
          <label className="mb-1 block text-sm font-medium text-gray-300">
            Recolor Method
          </label>
          <select
            value={method}
            onChange={(e) => setMethod(e.target.value as RecolorMethod)}
            className="w-full rounded-lg border border-gray-700 bg-gray-800 px-3 py-2 text-white"
          >
            <option value="reinhard">Reinhard — Natural color transfer</option>
            <option value="shift">Shift — Relative hue shift</option>
            <option value="overlay">Overlay — Simple tint (flat)</option>
          </select>
        </div>

        <ColorPicker value={color} onChange={setColor} />
        <ControlPanel
          intensity={intensity}
          lift={lift}
          onIntensityChange={setIntensity}
          onLiftChange={setLift}
        />

        <button
          onClick={handleProcess}
          disabled={!imageFile || segmenting}
          className="w-full rounded-lg bg-indigo-600 px-4 py-3 font-medium text-white transition hover:bg-indigo-500 disabled:opacity-40 disabled:cursor-not-allowed"
        >
          {segmenting ? "Detecting hair..." : hasMask ? "Re-detect Hair" : "Apply Color"}
        </button>

        {hasMask && (
          <p className="text-xs text-green-400">
            Hair detected — adjust controls above for live preview
          </p>
        )}

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
  );
}

function App() {
  const [activeTab, setActiveTab] = useState("color");

  return (
    <div className="min-h-screen bg-gray-950 text-white">
      <header className="border-b border-gray-800 px-6 py-4">
        <div className="mx-auto max-w-6xl flex items-center justify-between">
          <div>
            <h1 className="text-2xl font-bold">Hair Studio</h1>
            <p className="text-sm text-gray-400">
              AI-powered hair color and hairstyle tools
            </p>
          </div>
          <TabBar activeTab={activeTab} onChange={setActiveTab} />
        </div>
      </header>

      <main className="mx-auto max-w-6xl p-6">
        {activeTab === "color" && <ColorChangeTab />}
        {activeTab === "hairstyle" && <HairstyleGenerator />}
      </main>
    </div>
  );
}

export default App;
