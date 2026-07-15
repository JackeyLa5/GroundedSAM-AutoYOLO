import { useCallback } from "react";
import { useAppStore } from "@/store/useAppStore";
import { useDetectionListInfiniteQuery } from "./useDetection";
import { getDetection } from "@/services/api";
import { API_BASE } from "@/lib/constants";
import type { Detection } from "@/types";
import { parseCategories } from "@/lib/parsers";
import type { FilterMode } from "@/lib/filterBoxes";
import { batchFileMap } from "@/lib/cache";

export function useDetectionHistory() {
  const {
    setPreviewUrl,
    setCategories,
    setFilterMode,
    setNmsIou,
    setBoxCategoryFilter,
    setHiddenIndices,
    setResult,
    setFiles,
    setBatchResults,
  } = useAppStore();

  const historyQuery = useDetectionListInfiniteQuery();
  const allItems = historyQuery.data?.pages.flatMap((p) => p.items) ?? [];
  const total = historyQuery.data?.pages[0]?.total ?? 0;
  const recentCategories = Array.from<string>(
    new Set(allItems.flatMap((d: Detection) => parseCategories(d.categories))),
  ).sort();

  const handleSelectHistory = useCallback(
    async (det: Detection) => {
      setFiles([]);
      batchFileMap.clear();
      try {
        const full = await getDetection(det.id);
        setBatchResults([]);
        setResult(full);
        setPreviewUrl(`${API_BASE}/detections/${det.id}/image`);
        setCategories(parseCategories(full.categories));
        setFilterMode((full.filterMode as FilterMode) || "all");
        setBoxCategoryFilter([]);
        if (full.filterNmsIou != null) setNmsIou(full.filterNmsIou);
        setHiddenIndices(new Set());
      } catch (e) {
        console.error("Failed to load history detection:", e);
      }
    },
    [
      setBatchResults,
      setFiles,
      setPreviewUrl,
      setCategories,
      setFilterMode,
      setNmsIou,
      setBoxCategoryFilter,
      setHiddenIndices,
      setResult,
    ],
  );

  const handleShowHistoryBatch = useCallback(
    async (ids: string[]) => {
      if (ids.length === 0) return;
      setFiles([]);
      batchFileMap.clear();
      try {
        const results = await Promise.all(ids.map((id) => getDetection(id)));
        const first = results[0];
        setBatchResults(results);
        setResult(first);
        setPreviewUrl(`${API_BASE}/detections/${first.id}/image`);
        setCategories(parseCategories(first.categories));
        setFilterMode((first.filterMode as FilterMode) || "all");
        setBoxCategoryFilter([]);
        if (first.filterNmsIou != null) setNmsIou(first.filterNmsIou);
        setHiddenIndices(new Set());
      } catch (e) {
        console.error("Failed to load history batch:", e);
      }
    },
    [
      setBatchResults,
      setFiles,
      setPreviewUrl,
      setCategories,
      setFilterMode,
      setNmsIou,
      setBoxCategoryFilter,
      setHiddenIndices,
      setResult,
    ],
  );

  return {
    historyQuery,
    allItems,
    total,
    recentCategories,
    handleSelectHistory,
    handleShowHistoryBatch,
  };
}
