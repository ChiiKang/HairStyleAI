export type SegmentationModel = "bisenet" | "fashn" | "segformer";

export type RecolorMethod = "reinhard" | "shift" | "overlay";

export interface RecolorParams {
  color: string;
  intensity: number;
  lift: number;
  method: RecolorMethod;
}
