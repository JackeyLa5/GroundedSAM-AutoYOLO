import { create } from "zustand";
import type { FilterMode } from "@/lib/filterBoxes";

interface AppState {
  // Model Config
  appMode: "annotate" | "validate";
  setAppMode: (mode: "annotate" | "validate") => void;
  useGroundedSamSeg: boolean;
  setUseGroundedSamSeg: (v: boolean) => void;
  groundedSamThreshold: number;
  setGroundedSamThreshold: (v: number) => void;
  groundedSamMaskThreshold: number;
  setGroundedSamMaskThreshold: (v: number) => void;
  groundedSamText: string;
  setGroundedSamText: (v: string) => void;

  // Upload State
  inputMode: "image" | "video";
  setInputMode: (mode: "image" | "video") => void;
  files: File[];
  setFiles: (files: File[]) => void;
  previewUrl: string | null;
  setPreviewUrl: (url: string | null) => void;
  categories: string[];
  setCategories: (categories: string[]) => void;

  // Annotation State
  canvasMode: "view" | "draw";
  setCanvasMode: (mode: "view" | "draw") => void;
  drawCategory: string;
  setDrawCategory: (cat: string) => void;
  hiddenIndices: Set<string>;
  setHiddenIndices: (indices: Set<string> | ((prev: Set<string>) => Set<string>)) => void;
  filterMode: FilterMode;
  setFilterMode: (mode: FilterMode) => void;
  nmsIou: number;
  setNmsIou: (iou: number) => void;
  boxCategoryFilter: string[];
  setBoxCategoryFilter: (categories: string[]) => void;

  // Yolo Validation State
  validateModelSource: "trained" | "upload";
  setValidateModelSource: (source: "trained" | "upload") => void;
  selectedTrainedJobId: string | null;
  setSelectedTrainedJobId: (id: string | null) => void;
  validateVideoId: string | null;
  setValidateVideoId: (id: string | null) => void;
  validateRunKey: number;
  setValidateRunKey: (key: number | ((prev: number) => number)) => void;
  externalModelFile: File | null;
  setExternalModelFile: (file: File | null) => void;
  validateConf: number;
  setValidateConf: (conf: number) => void;
  validateIou: number;
  setValidateIou: (iou: number) => void;

  // Shared Detection State
  result: import("@/types").Detection | null;
  setResult: (result: import("@/types").Detection | null) => void;
  batchResults: import("@/types").Detection[];
  setBatchResults: (results: import("@/types").Detection[] | ((prev: import("@/types").Detection[]) => import("@/types").Detection[])) => void;

  // Training State
  isTraining: boolean;
  setIsTraining: (v: boolean) => void;
}

export const useAppStore = create<AppState>((set) => ({
  // Model Config
  appMode: "annotate",
  setAppMode: (mode) => set({ appMode: mode }),
  useGroundedSamSeg: true,
  setUseGroundedSamSeg: (useGroundedSamSeg) => set({ useGroundedSamSeg }),
  groundedSamThreshold: 0.5,
  setGroundedSamThreshold: (groundedSamThreshold) => set({ groundedSamThreshold }),
  groundedSamMaskThreshold: 0.5,
  setGroundedSamMaskThreshold: (groundedSamMaskThreshold) => set({ groundedSamMaskThreshold }),
  groundedSamText: "",
  setGroundedSamText: (groundedSamText) => set({ groundedSamText }),

  // Upload State
  inputMode: "image",
  setInputMode: (inputMode) => set({ inputMode }),
  files: [],
  setFiles: (files) => set({ files }),
  previewUrl: null,
  setPreviewUrl: (previewUrl) => set({ previewUrl }),
  categories: [],
  setCategories: (categories) => set({ categories }),

  // Annotation State
  canvasMode: "view",
  setCanvasMode: (canvasMode) => set({ canvasMode }),
  drawCategory: "",
  setDrawCategory: (drawCategory) => set({ drawCategory }),
  hiddenIndices: new Set(),
  setHiddenIndices: (updater) =>
    set((state) => ({
      hiddenIndices: typeof updater === "function" ? updater(state.hiddenIndices) : updater,
    })),
  filterMode: "all",
  setFilterMode: (filterMode) => set({ filterMode }),
  nmsIou: 0.5,
  setNmsIou: (nmsIou) => set({ nmsIou }),
  boxCategoryFilter: [],
  setBoxCategoryFilter: (boxCategoryFilter) => set({ boxCategoryFilter }),

  // Yolo Validation State
  validateModelSource: "trained",
  setValidateModelSource: (validateModelSource) => set({ validateModelSource }),
  selectedTrainedJobId: null,
  setSelectedTrainedJobId: (selectedTrainedJobId) => set({ selectedTrainedJobId }),
  validateVideoId: null,
  setValidateVideoId: (validateVideoId) => set({ validateVideoId }),
  validateRunKey: 0,
  setValidateRunKey: (updater) =>
    set((state) => ({
      validateRunKey: typeof updater === "function" ? updater(state.validateRunKey) : updater,
    })),
  externalModelFile: null,
  setExternalModelFile: (externalModelFile) => set({ externalModelFile }),
  validateConf: 0.25,
  setValidateConf: (validateConf) => set({ validateConf }),
  validateIou: 0.7,
  setValidateIou: (validateIou) => set({ validateIou }),

  // Shared Detection State
  result: null,
  setResult: (result) => set({ result }),
  batchResults: [],
  setBatchResults: (updater) =>
    set((state) => ({
      batchResults: typeof updater === "function" ? updater(state.batchResults) : updater,
    })),

  // Training State
  isTraining: false,
  setIsTraining: (isTraining) => set({ isTraining }),
}));
