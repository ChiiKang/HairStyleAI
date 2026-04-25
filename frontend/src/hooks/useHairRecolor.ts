import { useState, useCallback } from "react";
import type { SegmentationModel } from "../types";

interface HairRecolorState {
  maskUrl: string | null;
  resultUrl: string | null;
  loading: boolean;
  error: string | null;
}

export function useHairRecolor() {
  const [state, setState] = useState<HairRecolorState>({
    maskUrl: null,
    resultUrl: null,
    loading: false,
    error: null,
  });

  const processImage = useCallback(
    async (
      imageFile: File,
      model: SegmentationModel,
      color: string,
      intensity: number,
      lift: number
    ) => {
      // Revoke old URLs to prevent memory leaks
      setState((s) => {
        if (s.maskUrl) URL.revokeObjectURL(s.maskUrl);
        if (s.resultUrl) URL.revokeObjectURL(s.resultUrl);
        return { ...s, loading: true, error: null, maskUrl: null, resultUrl: null };
      });

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
        const maskUrl = URL.createObjectURL(maskBlob);

        const recolorForm = new FormData();
        recolorForm.append("image", imageFile);
        recolorForm.append("mask", maskBlob, "mask.png");
        recolorForm.append("color", color);
        recolorForm.append("intensity", String(intensity));
        recolorForm.append("lift", String(lift));

        const recolorRes = await fetch("/api/recolor", {
          method: "POST",
          body: recolorForm,
        });
        if (!recolorRes.ok) throw new Error(`Recoloring failed: ${recolorRes.statusText}`);

        const resultBlob = await recolorRes.blob();
        const resultUrl = URL.createObjectURL(resultBlob);

        setState({ maskUrl, resultUrl, loading: false, error: null });
      } catch (err) {
        setState((s) => ({
          ...s,
          loading: false,
          error: err instanceof Error ? err.message : "Unknown error",
        }));
      }
    },
    []
  );

  const reset = useCallback(() => {
    setState((s) => {
      if (s.maskUrl) URL.revokeObjectURL(s.maskUrl);
      if (s.resultUrl) URL.revokeObjectURL(s.resultUrl);
      return { maskUrl: null, resultUrl: null, loading: false, error: null };
    });
  }, []);

  return { ...state, processImage, reset };
}
