import { useCallback, useRef, useState, useEffect } from "react";

interface Props {
  onImageCaptured: (file: File, previewUrl: string) => void;
  previewUrl: string | null;
}

export function SelfieCapture({ onImageCaptured, previewUrl }: Props) {
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const inputRef = useRef<HTMLInputElement>(null);
  const [streaming, setStreaming] = useState(false);
  const [cameraRequested, setCameraRequested] = useState(false);
  const streamRef = useRef<MediaStream | null>(null);

  // Acquire the camera stream when requested
  useEffect(() => {
    if (!cameraRequested) return;

    let cancelled = false;

    (async () => {
      try {
        const stream = await navigator.mediaDevices.getUserMedia({
          video: { facingMode: "user", width: { ideal: 512 }, height: { ideal: 512 } },
        });
        if (cancelled) {
          stream.getTracks().forEach((t) => t.stop());
          return;
        }
        streamRef.current = stream;
        setStreaming(true);
      } catch {
        if (!cancelled) {
          setCameraRequested(false);
          alert("Could not access camera. Please upload a photo instead.");
        }
      }
    })();

    return () => {
      cancelled = true;
    };
  }, [cameraRequested]);

  // Attach stream to video element AFTER it's rendered (streaming = true)
  useEffect(() => {
    if (!streaming || !streamRef.current) return;

    const video = videoRef.current;
    if (!video) return;

    video.srcObject = streamRef.current;
    video.play().catch(() => {});
  }, [streaming]);

  const stopCamera = useCallback(() => {
    streamRef.current?.getTracks().forEach((t) => t.stop());
    streamRef.current = null;
    setStreaming(false);
    setCameraRequested(false);
  }, []);

  const takeSelfie = useCallback(() => {
    const video = videoRef.current;
    const canvas = canvasRef.current;
    if (!video || !canvas) return;

    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    // Mirror the selfie
    ctx.translate(canvas.width, 0);
    ctx.scale(-1, 1);
    ctx.drawImage(video, 0, 0);
    ctx.setTransform(1, 0, 0, 1, 0, 0);

    canvas.toBlob((blob) => {
      if (!blob) return;
      const file = new File([blob], "selfie.png", { type: "image/png" });
      const url = URL.createObjectURL(blob);
      onImageCaptured(file, url);
      stopCamera();
    }, "image/png");
  }, [onImageCaptured, stopCamera]);

  const handleUpload = useCallback(
    (file: File) => {
      const url = URL.createObjectURL(file);
      onImageCaptured(file, url);
      if (streaming) stopCamera();
    },
    [onImageCaptured, streaming, stopCamera]
  );

  const handleDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault();
      const file = e.dataTransfer.files[0];
      if (file && file.type.startsWith("image/")) handleUpload(file);
    },
    [handleUpload]
  );

  if (previewUrl) {
    return (
      <div className="space-y-3">
        <img
          src={previewUrl}
          alt="Selected"
          className="mx-auto max-h-72 rounded-xl object-contain"
        />
        <button
          onClick={() => onImageCaptured(null as unknown as File, "")}
          className="w-full rounded-lg border border-gray-700 px-3 py-2 text-sm text-gray-400 hover:bg-gray-800 transition"
        >
          Change Photo
        </button>
      </div>
    );
  }

  return (
    <div className="space-y-3">
      {streaming ? (
        <div className="space-y-3">
          <video
            ref={videoRef}
            className="mx-auto max-h-72 rounded-xl object-contain"
            style={{ transform: "scaleX(-1)" }}
            autoPlay
            muted
            playsInline
          />
          <div className="flex gap-2">
            <button
              onClick={takeSelfie}
              className="flex-1 rounded-lg bg-indigo-600 px-4 py-2 text-sm font-medium text-white hover:bg-indigo-500 transition"
            >
              Take Selfie
            </button>
            <button
              onClick={stopCamera}
              className="rounded-lg border border-gray-700 px-4 py-2 text-sm text-gray-400 hover:bg-gray-800 transition"
            >
              Cancel
            </button>
          </div>
        </div>
      ) : (
        <div
          onDrop={handleDrop}
          onDragOver={(e) => e.preventDefault()}
          className="rounded-xl border-2 border-dashed border-gray-700 p-6 text-center space-y-4"
        >
          <p className="text-gray-400">Upload a photo or take a selfie</p>
          <div className="flex gap-3 justify-center">
            <button
              onClick={() => setCameraRequested(true)}
              disabled={cameraRequested}
              className="rounded-lg bg-indigo-600 px-4 py-2 text-sm font-medium text-white hover:bg-indigo-500 transition disabled:opacity-50"
            >
              {cameraRequested ? "Opening..." : "Open Camera"}
            </button>
            <button
              onClick={() => inputRef.current?.click()}
              className="rounded-lg border border-gray-700 px-4 py-2 text-sm text-gray-300 hover:bg-gray-800 transition"
            >
              Upload Photo
            </button>
          </div>
          <p className="text-xs text-gray-500">JPG, PNG supported</p>
        </div>
      )}
      <canvas ref={canvasRef} className="hidden" />
      <input
        ref={inputRef}
        type="file"
        accept="image/*"
        className="hidden"
        onChange={(e) => {
          const file = e.target.files?.[0];
          if (file) handleUpload(file);
        }}
      />
    </div>
  );
}
