import { Dropdown, Modal } from "antd";
import { getFileUrl } from "@/lib/cache";
import { API_BASE } from "@/lib/constants";

interface Props {
  result: Detection | null;
  previewUrl: string;
  batchResults: Detection[];
  batchFiles: File[];
  loading: boolean;
  elapsedMs: number;
  categories: string[];
  canvasMode: "view" | "draw";
  drawCategory: string;
  recentCategories: string[];
  hiddenIndices: Set<string>;
  onToggleVisibility: (boxId: string) => void;
  onCanvasModeChange: (mode: "view" | "draw") => void;
  onDrawCategoryChange: (cat: string) => void;
  onDeleteBox: (boxId: string) => void;
  onSelectBatch: (det: Detection, file?: File) => void;
  onSelectPending?: (url: string) => void;
  onReDetect: () => void;
  onReDetectAll?: () => void;
  onSaveBoxes: () => void;
  onDrawBox: (box: { x1: number; y1: number; x2: number; y2: number }) => void;
  isValidation?: boolean;
  isRedetecting?: boolean;
  maskingBox?: boolean;
}

const EMPTY_BOXES: never[] = [];

export function DetectionResult({
  result,
  previewUrl,
  batchResults,
  batchFiles,
  loading,
  elapsedMs,
  categories,
  canvasMode,
  drawCategory,
  recentCategories,
  hiddenIndices,
  onToggleVisibility,
  onCanvasModeChange,
  onDrawCategoryChange,
  onDeleteBox,
  onSelectBatch,
  onSelectPending,
  onReDetect,
  onReDetectAll,
  onSaveBoxes,
  onDrawBox,
  isValidation = false,
  isRedetecting = false,
  maskingBox = false,
}: Props) {
  const { t } = useTranslation();

  const batchItemCount = Math.max(batchFiles.length, batchResults.length);
  const batchPreviewUrls = Array.from({ length: batchItemCount }, (_, i) =>
    batchFiles[i]
      ? getFileUrl(batchFiles[i])
      : batchResults[i]
        ? `${API_BASE}/detections/${batchResults[i].id}/image`
        : "",
  );
  const exportWithZeroTargetCheck = (ids: string[], format: string, label: string) => {
    const selectedResults =
      batchResults.length > 1
        ? batchResults.filter((r) => ids.includes(r.id))
        : result
          ? [result]
          : [];
    const zero = selectedResults.filter((r) => r.boxes.length === 0);
    const positiveIds = selectedResults.filter((r) => r.boxes.length > 0).map((r) => r.id);
    const runExport = async (targetIds: string[]) => {
      const blob = await exportBatch(targetIds, format);
      downloadBlob(blob, `${label}_dataset.zip`);
    };
    if (positiveIds.length === 0) {
      toast.error(t("trainingPanel.noPositiveTargets"));
      return;
    }
    if (zero.length === 0) {
      runExport(ids);
      return;
    }
    Modal.confirm({
      title: t("trainingPanel.zeroTargetWarningTitle"),
      content: t("trainingPanel.zeroTargetWarningMessage", {
        count: zero.length,
        action: t("trainingPanel.zeroTargetAction.export"),
      }),
      okText: t("trainingPanel.ignoreAndContinue"),
      cancelText: t("common.cancel"),
      okButtonProps: { danger: true },
      onOk: () => runExport(positiveIds),
    });
  };

  return (
    <div className="space-y-4">
      <div className="relative">
        <DetectionCanvas
          imageUrl={previewUrl}
          boxes={result?.boxes || EMPTY_BOXES}
          imgWidth={result?.imageWidth || 0}
          imgHeight={result?.imageHeight || 0}
          mode={canvasMode}
          hiddenIndices={hiddenIndices}
          onModeChange={onCanvasModeChange}
          onDrawBox={onDrawBox}
        />
        {canvasMode === "draw" && result && (
          <div className="mt-2 flex items-center gap-2">
            <input
              type="text"
              value={drawCategory}
              onChange={(e) => onDrawCategoryChange(e.target.value)}
              placeholder={t("detectionResult.drawCategoryPlaceholder")}
              className="rounded border border-gray-300 px-2 py-1 text-xs w-32"
            />
            {recentCategories.map((c) => (
              <button
                key={c}
                type="button"
                onClick={() => onDrawCategoryChange(c)}
                className={`rounded px-1.5 py-0.5 text-xs font-medium transition-colors ${
                  drawCategory === c
                    ? "bg-primary-500 text-white"
                    : "bg-gray-100 text-gray-500 hover:bg-gray-200"
                }`}
              >
                {c}
              </button>
            ))}
          </div>
        )}
        {loading && (!result || isRedetecting) && <LoadingOverlay elapsedMs={elapsedMs} />}
        {maskingBox && (
          <LoadingOverlay
            elapsedMs={0}
            title={t("detectionResult.samMasking")}
            subtitle={t("detectionResult.pleaseWait")}
          />
        )}
      </div>

      <div className="flex items-center justify-between">
        <h2 className="text-sm font-medium text-gray-700">
          {result 
            ? t("detectionResult.resultCount", { count: result.boxes.length }) 
            : loading 
              ? t("common.detecting") 
              : ""}
          {result?.elapsedMs != null && result.elapsedMs > 0 && (
            <span className="ml-2 text-gray-400 font-normal">
              {t("detectionResult.timeElapsed")}{" "}
              {result.elapsedMs >= 1000
                ? `${(result.elapsedMs / 1000).toFixed(1)}s`
                : `${result.elapsedMs}ms`}
            </span>
          )}
          {result && batchResults.length > 1 && (
            <span className="ml-2 text-gray-400 font-normal">
              — {result.imageName} ({batchResults.indexOf(result) + 1}/{batchResults.length})
            </span>
          )}
        </h2>

        <div className="flex gap-2">
          {!isValidation && (
            <button
              type="button"
              disabled={!result || loading}
              onClick={onSaveBoxes}
              className="rounded border border-green-300 px-3 py-1 text-xs font-medium text-green-600 hover:bg-green-50 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
            >
              {t("detectionResult.saveFilter")}
            </button>
          )}
          {!isValidation && categories.length > 0 && (
            <>
              <button
                type="button"
                disabled={!result || loading}
                onClick={onReDetect}
                className="rounded bg-orange-500 px-3 py-1 text-xs font-medium text-white hover:bg-orange-600 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
              >
                {loading ? t("common.detecting") : t("detectionResult.redetectCurrent")}
              </button>
              {batchResults.length > 1 && onReDetectAll && (
                <button
                  type="button"
                  disabled={!result || loading}
                  onClick={onReDetectAll}
                  className="rounded border border-orange-300 px-3 py-1 text-xs font-medium text-orange-600 hover:bg-orange-50 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
                >
                  {loading ? t("common.detecting") : t("detectionResult.redetectAll")}
                </button>
              )}
            </>
          )}
          <Dropdown
            disabled={!result || loading}
            menu={{
              items: [
                { key: "yolo", label: "YOLO (.txt)" },
                { key: "yolo-seg", label: "YOLO Segmentation" },
                { key: "coco", label: "COCO (.json)" },
                { key: "voc", label: "Pascal VOC (.xml)" },
                { key: "createml", label: "CreateML (.json)" },
              ],
              onClick: async ({ key }) => {
                if (!result) return;
                if (key === "yolo") {
                  downloadYoloTxt(
                    result.boxes,
                    categories,
                    result.imageWidth,
                    result.imageHeight,
                    result.imageName,
                  );
                } else {
                  const labels: Record<string, string> = {
                    coco: "COCO",
                    "yolo-seg": "YOLO_Seg",
                    voc: "VOC",
                    createml: "CreateML",
                  };
                  const blob = await exportBatch([result.id], key);
                  downloadBlob(blob, `${labels[key] ?? key}_dataset.zip`);
                }
              },
            }}
            trigger={["click"]}
          >
            <button
              type="button"
              disabled={!result || loading}
              className="rounded border border-primary-200 px-3 py-1 text-xs font-medium text-primary-600 hover:bg-primary-50 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
            >
              {t("detectionResult.exportLabel")}
            </button>
          </Dropdown>
          {!isValidation && (
            <Dropdown
              disabled={!result || loading}
              menu={{
                items: [
                  { key: "yolo", label: "YOLO (.txt)" },
                  { key: "yolo-seg", label: "YOLO Segmentation" },
                  { key: "coco", label: "COCO (.json)" },
                  { key: "voc", label: "Pascal VOC (.xml)" },
                  { key: "createml", label: "CreateML (.json)" },
                ],
                onClick: async ({ key }) => {
                  if (!result) return;
                  const ids = batchResults.length > 1 ? batchResults.map((r) => r.id) : [result.id];
                  const labels: Record<string, string> = {
                    yolo: "YOLO",
                    "yolo-seg": "YOLO_Seg",
                    coco: "COCO",
                    voc: "VOC",
                    createml: "CreateML",
                  };
                  exportWithZeroTargetCheck(ids, key, labels[key] ?? key);
                },
              }}
              trigger={["click"]}
            >
              <button
                type="button"
                disabled={!result || loading}
                className="rounded bg-primary-600 px-3 py-1 text-xs font-medium text-white hover:bg-primary-700 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
              >
                {batchResults.length > 1
                  ? t("detectionResult.exportAllDataset", { count: batchResults.length })
                  : t("detectionResult.exportDataset")}
              </button>
            </Dropdown>
          )}
        </div>
      </div>

      {batchItemCount > 1 && (
        <div className="flex gap-2 overflow-x-auto pb-2">
          {Array.from({ length: batchItemCount }, (_, i) => {
            const file = batchFiles[i];
            const res = batchResults[i];
            const done = !!res;
            const pending = !done && loading;
            const preview = batchPreviewUrls[i];
            const anyPendingSelected = batchPreviewUrls.some(
              (url, idx) => !batchResults[idx] && previewUrl === url,
            );
            const isActive = anyPendingSelected
              ? previewUrl === preview
              : done && result?.id === res.id;
            return (
              <button
                key={i}
                onClick={() => {
                  if (res) {
                    onSelectBatch(res, file);
                  } else {
                    onSelectPending?.(preview);
                  }
                }}
                disabled={false}
                className={`relative flex-shrink-0 rounded border-2 p-1 transition-colors ${
                  isActive
                    ? "border-primary-500"
                    : done
                      ? "border-transparent hover:border-gray-200"
                      : "border-gray-200 opacity-40"
                }`}
              >
                <img
                  src={preview}
                  alt={file?.name ?? res?.imageName ?? ""}
                  className="h-14 w-14 rounded object-cover"
                />
                {done ? (
                  <span className="absolute -top-1 -right-1 bg-primary-500 text-white text-[10px] font-bold rounded-full w-4 h-4 flex items-center justify-center leading-none">
                    {res.boxes.length}
                  </span>
                ) : pending ? (
                  <div className="absolute inset-0 bg-white/50 rounded flex items-center justify-center">
                    <svg className="animate-spin h-4 w-4 text-primary-500" viewBox="0 0 24 24">
                      <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" fill="none" />
                      <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
                    </svg>
                  </div>
                ) : null}
              </button>
            );
          })}
        </div>
      )}

      <ResultTable
        boxes={result?.boxes || EMPTY_BOXES}
        hiddenIndices={hiddenIndices}
        onToggleVisibility={onToggleVisibility}
        onDelete={onDeleteBox}
      />
    </div>
  );
}

function LoadingOverlay({
  elapsedMs,
  title,
  subtitle,
}: {
  elapsedMs: number;
  title?: string;
  subtitle?: string;
}) {
  const { t } = useTranslation();
  const { groundedSam } = useModelEvents();
  const modelLoading =
    groundedSam.status === "loading" ||
    groundedSam.status === "starting";

  return (
    <div className="absolute inset-0 bg-white/60 rounded-lg flex flex-col items-center justify-center gap-3">
      <svg className="animate-spin h-10 w-10 text-primary-500" viewBox="0 0 24 24">
        <circle
          className="opacity-25"
          cx="12"
          cy="12"
          r="10"
          stroke="currentColor"
          strokeWidth="4"
          fill="none"
        />
        <path
          className="opacity-75"
          fill="currentColor"
          d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z"
        />
      </svg>
      <p className="text-sm font-medium text-gray-600">
        {title ?? (modelLoading ? t("detectionResult.loadingModel") : t("detectionResult.detecting"))}
      </p>
      <p className="text-xs text-gray-400">
        {subtitle ?? (modelLoading ? t("detectionResult.pleaseWait") : `${(elapsedMs / 1000).toFixed(1)}s`)}
      </p>
    </div>
  );
}
