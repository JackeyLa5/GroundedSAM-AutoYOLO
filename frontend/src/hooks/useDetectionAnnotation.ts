import { useCallback, useMemo, useState } from "react";
import toast from "react-hot-toast";
import { useQueryClient, type InfiniteData } from "@tanstack/react-query";
import { useTranslation } from "react-i18next";
import { useAppStore } from "@/store/useAppStore";
import { addBox, deleteBox, saveFilterSettings } from "@/services/api";
import { applyFilter } from "@/lib/filterBoxes";
import type { Detection } from "@/types";

type DetectionListPage = { items: Detection[]; total: number };
type DetectionListData = InfiniteData<DetectionListPage, number>;

export function useDetectionAnnotation() {
  const { t } = useTranslation();
  const queryClient = useQueryClient();
  const { drawCategory, filterMode, nmsIou, boxCategoryFilter, setHiddenIndices, result } =
    useAppStore();
  const [maskingBox, setMaskingBox] = useState(false);

  const syncDetection = useCallback(
    (updated: Detection) => {
      useAppStore.setState((state) => ({
        result: state.result?.id === updated.id ? updated : state.result,
        batchResults: state.batchResults.map((r) => (r.id === updated.id ? updated : r)),
      }));
      queryClient.setQueryData<DetectionListData>(["detections"], (data) => {
        if (!data) return data;
        return {
          ...data,
          pages: data.pages.map((page) => ({
            ...page,
            items: page.items.map((det) => (det.id === updated.id ? updated : det)),
          })),
        };
      });
    },
    [queryClient],
  );

  const handleDrawBox = useCallback(
    async (raw: { x1: number; y1: number; x2: number; y2: number }) => {
      if (!result || !drawCategory.trim()) {
        toast.error(t("home.drawCategoryRequired"));
        return;
      }
      try {
        setMaskingBox(true);
        const newBox = await addBox(result.id, { ...raw, className: drawCategory.trim() });
        const latest = useAppStore.getState().result;
        if (!latest || latest.id !== result.id) return;
        const exists = latest.boxes.some((box) => box.id === newBox.id);
        const updated = { ...latest, boxes: exists ? latest.boxes : [...latest.boxes, newBox] };
        syncDetection(updated);
      } catch (e) {
        console.error("Draw box failed:", e);
        toast.error(t("home.drawBoxFailed") || "Failed to save box");
      } finally {
        setMaskingBox(false);
      }
    },
    [result, drawCategory, syncDetection, t],
  );

  const handleDeleteBox = useCallback(
    async (boxId: string) => {
      const latest = useAppStore.getState().result;
      if (!latest) return;
      const box = latest.boxes.find((b) => b.id === boxId);
      if (!box) return;
      try {
        await deleteBox(latest.id, box.id);
        const current = useAppStore.getState().result;
        if (!current || current.id !== latest.id) return;
        const updated = { ...current, boxes: current.boxes.filter((b) => b.id !== boxId) };
        syncDetection(updated);
      } catch (e) {
        console.error("Delete box failed:", e);
        toast.error(t("home.deleteBoxFailed") || "Failed to delete box");
      }
    },
    [syncDetection, t],
  );

  const handleSaveBoxes = useCallback(async () => {
    if (!result) return;

    try {
      await saveFilterSettings(result.id, filterMode, filterMode === "nms" ? nmsIou : null);
      toast.success(t("home.savedSuccessfully"));
      syncDetection({
        ...result,
        filterMode: filterMode,
        filterNmsIou: filterMode === "nms" ? nmsIou : null,
      });
    } catch (e) {
      console.error("Save boxes failed:", e);
      toast.error(t("home.saveFailed"));
    }
  }, [result, filterMode, nmsIou, syncDetection, t]);

  const toggleBoxVisibility = useCallback(
    (id: string) => {
      setHiddenIndices((prev) => {
        const next = new Set(prev);
        if (next.has(id)) next.delete(id);
        else next.add(id);
        return next;
      });
    },
    [setHiddenIndices],
  );

  const displayResult = useMemo(
    () => {
      if (!result) return null;
      const categorySet = new Set(boxCategoryFilter);
      const categoryBoxes =
        categorySet.size > 0
          ? result.boxes.filter((box) => categorySet.has(box.className))
          : result.boxes;
      return { ...result, boxes: applyFilter(filterMode, categoryBoxes, nmsIou) };
    },
    [result, filterMode, nmsIou, boxCategoryFilter],
  );

  return {
    handleDrawBox,
    handleDeleteBox,
    handleSaveBoxes,
    toggleBoxVisibility,
    displayResult,
    maskingBox,
  };
}
