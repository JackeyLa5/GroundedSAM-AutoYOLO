import { ErrorBoundary } from "@/components/ErrorBoundary";
import { ImageUploader } from "@/components/ImageUploader";
import { HistoryList } from "@/components/HistoryList";
import { BatchProgress } from "@/components/BatchProgress";
import { GroundedSamStatus } from "@/components/GroundedSamStatus";
import { TrainingPanel } from "@/components/TrainingPanel";
import { VideoPanel } from "@/components/VideoPanel";
import { ValidationSettings } from "@/components/ValidationSettings";
import { FilterPanel } from "@/components/FilterPanel";
import { HistorySkeleton } from "@/components/LoadingSkeleton";
import { SidebarHeader } from "./Header";
import { DetectionControls } from "./DetectionControls";
import type { Detection } from "@/types";
import { useAppStore } from "@/store/useAppStore";

export interface SidebarProps {
  recentCategories: string[];
  handleFiles: (fs: File[]) => void;
  handleDetect: () => void;
  handleSelectHistory: (det: Detection) => void;
  handleShowHistoryBatch: (ids: string[]) => void;
  loading: boolean;
  batchProgress: { current: number; total: number };
  batchResults: Detection[];
  cancel: () => void;
  historyQuery: { hasNextPage: boolean; isFetchingNextPage: boolean; fetchNextPage: () => unknown };
  allItems: Detection[];
  total: number;
  result: Detection | null;
  setResult: (result: Detection | null) => void;
  handleSelectKeyframe: (files: File[]) => void;
}

export function Sidebar({
  recentCategories,
  handleFiles,
  handleDetect,
  handleSelectHistory,
  handleShowHistoryBatch,
  loading,
  batchProgress,
  batchResults,
  cancel,
  historyQuery,
  allItems,
  total,
  result,
  setResult,
  handleSelectKeyframe,
}: SidebarProps) {
  const { t } = useTranslation();
  const {
    appMode, setAppMode,
    validateModelSource, setValidateModelSource,
    selectedTrainedJobId, setSelectedTrainedJobId,
    inputMode, setInputMode,
    files, setFiles,
    setPreviewUrl,
    setBatchResults: setBatch,
    validateVideoId, setValidateVideoId,
    setValidateRunKey,
    externalModelFile, setExternalModelFile,
    validateConf, setValidateConf,
    validateIou, setValidateIou,
    filterMode, setFilterMode,
    nmsIou, setNmsIou,
    boxCategoryFilter, setBoxCategoryFilter,
    setHiddenIndices,
  } = useAppStore();
  const resultCategories = useMemo(
    () => Array.from(new Set((result?.boxes ?? []).map((box) => box.className))).sort(),
    [result],
  );

  return (
    <aside
      className="flex-shrink-0 border-r border-gray-200 bg-white flex flex-col gap-4 overflow-y-auto relative"
      style={{ width: 440, padding: "1.25rem" }}
    >
      <SidebarHeader />

      <GroundedSamStatus />

      {/* Mode tabs */}
      <div className="flex border-b border-gray-200 mb-2">
        {(["annotate", "validate"] as const).map((m) => (
          <button
            key={m}
            onClick={() => { setAppMode(m); setResult(null); setValidateVideoId(null); }}
            className={`flex-1 pb-3 pt-1 text-sm font-bold transition-all duration-200 relative text-center cursor-pointer ${
              appMode === m
                ? "text-primary-600 border-b-2 border-primary-600 scale-[1.02]"
                : "text-gray-400 hover:text-gray-600 border-b-2 border-transparent"
            }`}
          >
            {{ annotate: t("common.annotate"), validate: t("common.validate") }[m]}
          </button>
        ))}
      </div>

      {appMode === "validate" && (
        <ValidationSettings
          selectedJobId={selectedTrainedJobId}
          onSelectJob={setSelectedTrainedJobId}
          modelSource={validateModelSource}
          onSourceChange={setValidateModelSource}
          externalFile={externalModelFile}
          onExternalFile={setExternalModelFile}
          validateConf={validateConf}
          onConfChange={setValidateConf}
          validateIou={validateIou}
          onIouChange={setValidateIou}
        />
      )}

      {/* Input mode: Image / Video */}
      <div>
        <div className="flex gap-1 rounded bg-gray-100 p-0.5 mb-2">
          {(["image", "video"] as const).map((mode) => (
            <button
              key={mode}
              onClick={() => { setInputMode(mode); setFiles([]); setPreviewUrl(null); setBatch([]); }}
              className={`flex-1 rounded px-3 py-1.5 text-xs font-medium transition-colors ${
                inputMode === mode
                  ? "bg-white text-primary-600 shadow-sm"
                  : "text-gray-500 hover:text-gray-700"
              }`}
            >
              {{ image: t("common.image"), video: t("common.video") }[mode]}
            </button>
          ))}
        </div>
        {inputMode === "image" ? (
          <ImageUploader
            onFiles={handleFiles}
            onClear={() => { setFiles([]); setPreviewUrl(null); setBatch([]); }}
            disabled={loading}
          />
        ) : (
          <VideoPanel
            onLoadKeyframes={handleSelectKeyframe}
            onValidateVideo={
              appMode === "validate"
                ? (videoId) => { setValidateVideoId(videoId); setValidateRunKey((k) => k + 1); }
                : undefined
            }
            disabled={loading}
          />
        )}
      </div>

      {!(appMode === "validate" && inputMode === "video" && validateVideoId) && (
        <DetectionControls
          recentCategories={recentCategories}
          loading={loading}
          filesCount={files.length}
          batchProgress={batchProgress}
          onDetect={handleDetect}
        />
      )}

      {appMode === "validate" && inputMode === "video" && (
        <div className="text-xs text-gray-400 text-center py-2">
          {validateVideoId ? t("home.placeholderVideoRunning") : t("home.placeholderSelectVideo")}
        </div>
      )}

      <BatchProgress
        current={batchProgress.current}
        total={batchProgress.total}
        completed={batchResults.length}
        onCancel={cancel}
      />

      {result && (
        <FilterPanel
          filterMode={filterMode}
          onFilterModeChange={setFilterMode}
          nmsIou={nmsIou}
          onNmsIouChange={setNmsIou}
          categories={resultCategories}
          selectedCategories={boxCategoryFilter}
          onSelectedCategoriesChange={setBoxCategoryFilter}
          setHiddenIndices={setHiddenIndices}
        />
      )}

      <hr className="border-gray-100" />

      <div>
        <p className="text-sm font-medium text-gray-600 mb-2">{t("common.history")}</p>
        <Suspense fallback={<HistorySkeleton />}>
          <ErrorBoundary>
            <HistoryList
              allItems={allItems}
              total={total}
              hasNextPage={historyQuery.hasNextPage}
              isFetchingNextPage={historyQuery.isFetchingNextPage}
              fetchNextPage={historyQuery.fetchNextPage}
              onSelect={handleSelectHistory}
              onShowSelected={handleShowHistoryBatch}
            />
          </ErrorBoundary>
        </Suspense>
      </div>

      <hr className="border-gray-100" />

      <div>
        <p className="text-sm font-medium text-gray-600 mb-2">{t("common.yoloTrain")}</p>
        <TrainingPanel
          detections={allItems}
          total={total}
          hasNextPage={historyQuery.hasNextPage}
          isFetchingNextPage={historyQuery.isFetchingNextPage}
          fetchNextPage={historyQuery.fetchNextPage}
        />
      </div>
    </aside>
  );
}
