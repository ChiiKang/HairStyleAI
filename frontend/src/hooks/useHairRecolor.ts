import { useState, useCallback, useRef, useEffect } from "react";
import type { SegmentationModel } from "../types";

interface HairRecolorState {
  maskUrl: string | null;
  resultUrl: string | null;
  segmenting: boolean;
  recoloring: boolean;
  error: string | null;
}

export function useHairRecolor() {
  const [state, setState] = useState<HairRecolorState>({
    maskUrl: null,
    resultUrl: null,
    segmenting: false,
    recoloring: false,
    error: null,
  });

  // Cache the mask blob and image file so recolor can reuse them
  const maskBlobRef = useRef<Blob | null>(null);
  const imageFileRef = useRef<File | null>(null);
  const recolorAbortRef = useRef<AbortController | null>(null);

  const segment = useCallback(
    async (imageFile: File, model: SegmentationModel) => {
      // Clean up old state
      setState((s) => {
        if (s.maskUrl) URL.revokeObjectURL(s.maskUrl);
        if (s.resultUrl) URL.revokeObjectURL(s.resultUrl);
        return { maskUrl: null, resultUrl: null, segmenting: true, recoloring: false, error: null };
      });
      maskBlobRef.current = null;
      imageFileRef.current = imageFile;

      try {
        const segForm = new FormData();
        segForm.append("image", imageFile);
        segForm.append("model", model);

        const segRes = await fetch("/api/segment", {
          method: "POST",
          body: segForm,
        });
        if (!segRes.ok) throw new Error(`Segmentation failed: ${segRes.statusText}`);

        const maskBlob = await segRes.blob();
        maskBlobRef.current = maskBlob;
        const maskUrl = URL.createObjectURL(maskBlob);

        setState((s) => ({ ...s, maskUrl, segmenting: false }));
        return true;
      } catch (err) {
        setState((s) => ({
          ...s,
          segmenting: false,
          error: err instanceof Error ? err.message : "Unknown error",
        }));
        return false;
      }
    },
    []
  );

  const recolor = useCallback(
    async (color: string, intensity: number, lift: number, method: string) => {
      if (!maskBlobRef.current || !imageFileRef.current) return;

      // Abort any in-flight recolor request
      recolorAbortRef.current?.abort();
      const controller = new AbortController();
      recolorAbortRef.current = controller;

      setState((s) => {
        if (s.resultUrl) URL.revokeObjectURL(s.resultUrl);
        return { ...s, recoloring: true, resultUrl: null, error: null };
      });

      try {
        const form = new FormData();
        form.append("image", imageFileRef.current);
        form.append("mask", maskBlobRef.current, "mask.png");
        form.append("color", color);
        form.append("intensity", String(intensity));
        form.append("lift", String(lift));
        form.append("method", method);

        const res = await fetch("/api/recolor", {
          method: "POST",
          body: form,
          signal: controller.signal,
        });
        if (!res.ok) throw new Error(`Recoloring failed: ${res.statusText}`);

        const resultBlob = await res.blob();
        const resultUrl = URL.createObjectURL(resultBlob);

        setState((s) => ({ ...s, resultUrl, recoloring: false }));
      } catch (err) {
        if (err instanceof DOMException && err.name === "AbortError") return;
        setState((s) => ({
          ...s,
          recoloring: false,
          error: err instanceof Error ? err.message : "Unknown error",
        }));
      }
    },
    []
  );

  const hasMask = maskBlobRef.current !== null;

  const reset = useCallback(() => {
    recolorAbortRef.current?.abort();
    maskBlobRef.current = null;
    imageFileRef.current = null;
    setState((s) => {
      if (s.maskUrl) URL.revokeObjectURL(s.maskUrl);
      if (s.resultUrl) URL.revokeObjectURL(s.resultUrl);
      return { maskUrl: null, resultUrl: null, segmenting: false, recoloring: false, error: null };
    });
  }, []);

  return { ...state, loading: state.segmenting || state.recoloring, hasMask, segment, recolor, reset };
}
