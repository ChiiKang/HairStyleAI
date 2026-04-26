import { useState, useCallback } from "react";

interface GeneratedResult {
  imageUrl: string | null;
  label: string;
}

interface HairstyleGenState {
  results: GeneratedResult[];
  loading: boolean;
  error: string | null;
  model: string;
  durationMs: number | null;
}

export function useHairstyleGenerator() {
  const [state, setState] = useState<HairstyleGenState>({
    results: [],
    loading: false,
    error: null,
    model: "",
    durationMs: null,
  });

  const generate = useCallback(async (imageFile: File, model: string) => {
    // Revoke old URLs
    setState((s) => {
      s.results.forEach((r) => {
        if (r.imageUrl) URL.revokeObjectURL(r.imageUrl);
      });
      return { results: [], loading: true, error: null, model, durationMs: null };
    });

    try {
      const form = new FormData();
      form.append("image", imageFile);
      form.append("model", model);

      const res = await fetch("/api/generate-hairstyles", {
        method: "POST",
        body: form,
      });

      if (!res.ok) {
        const detail = await res.text().catch(() => res.statusText);
        throw new Error(`Generation failed: ${detail}`);
      }

      const data = await res.json();

      // Backend returns { images: [...base64 or urls], labels: [...], model, duration_ms }
      const results: GeneratedResult[] = (data.images || []).map(
        (img: string, i: number) => {
          // Handle both base64 and URL responses
          const isBase64 = img.startsWith("data:") || !img.startsWith("http");
          const imageUrl = isBase64 && !img.startsWith("data:")
            ? `data:image/png;base64,${img}`
            : img;

          return {
            imageUrl,
            label: data.labels?.[i] || `Style ${i + 1}`,
          };
        }
      );

      setState({
        results,
        loading: false,
        error: null,
        model: data.model || model,
        durationMs: data.duration_ms ?? null,
      });
    } catch (err) {
      setState((s) => ({
        ...s,
        loading: false,
        error: err instanceof Error ? err.message : "Unknown error",
      }));
    }
  }, []);

  const reset = useCallback(() => {
    setState((s) => {
      s.results.forEach((r) => {
        if (r.imageUrl?.startsWith("blob:")) URL.revokeObjectURL(r.imageUrl);
      });
      return { results: [], loading: false, error: null, model: "", durationMs: null };
    });
  }, []);

  return { ...state, generate, reset };
}
