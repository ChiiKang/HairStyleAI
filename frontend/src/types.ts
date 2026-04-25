// frontend/src/types.ts
export type SegmentationModel = "bisenet" | "fashn" | "segformer";

export interface RecolorParams {
  color: string;
  intensity: number;
  lift: number;
}
