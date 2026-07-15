import { useEffect, useState } from "react";
import { Popconfirm } from "antd";

interface Props {
  boxes: BBox[];
  hiddenIndices: Set<string>;
  onToggleVisibility: (boxId: string) => void;
  onDelete?: (boxId: string) => void | Promise<void>;
}

export function ResultTable({ boxes, hiddenIndices, onToggleVisibility, onDelete }: Props) {
  const { t } = useTranslation();
  const [selectedIds, setSelectedIds] = useState<Set<string>>(new Set());
  const [deletingSelected, setDeletingSelected] = useState(false);

  useEffect(() => {
    const valid = new Set(boxes.map((box) => box.id));
    setSelectedIds((prev) => new Set([...prev].filter((id) => valid.has(id))));
  }, [boxes]);

  const allSelected = boxes.length > 0 && boxes.every((box) => selectedIds.has(box.id));
  const toggleAll = () => {
    setSelectedIds(allSelected ? new Set() : new Set(boxes.map((box) => box.id)));
  };
  const toggleOne = (id: string) => {
    setSelectedIds((prev) => {
      const next = new Set(prev);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return next;
    });
  };
  const deleteSelected = async () => {
    if (!onDelete || selectedIds.size === 0) return;
    setDeletingSelected(true);
    try {
      for (const id of [...selectedIds]) {
        await onDelete(id);
      }
      setSelectedIds(new Set());
    } finally {
      setDeletingSelected(false);
    }
  };

  if (boxes.length === 0) {
    return <p className="py-4 text-sm text-gray-400 text-center">{t("resultTable.noTargets")}</p>;
  }

  return (
    <div className="overflow-x-auto rounded-lg border border-gray-200">
      {onDelete && (
        <div className="flex items-center justify-between border-b border-gray-100 bg-gray-50 px-3 py-2 text-xs">
          <label className="flex items-center gap-2 text-gray-500">
            <input
              type="checkbox"
              checked={allSelected}
              onChange={toggleAll}
              className="h-3.5 w-3.5 rounded border-gray-300"
            />
            {t("resultTable.selectedBoxes", { count: selectedIds.size })}
          </label>
          {selectedIds.size > 0 && (
            <Popconfirm
              title={t("resultTable.deleteSelectedBoxesConfirm", { count: selectedIds.size })}
              onConfirm={deleteSelected}
              okText={t("common.delete")}
              cancelText={t("common.cancel")}
              okButtonProps={{ danger: true }}
            >
              <button
                type="button"
                disabled={deletingSelected}
                className="text-red-500 hover:text-red-600 disabled:opacity-50"
              >
                {t("resultTable.deleteSelectedBoxes", { count: selectedIds.size })}
              </button>
            </Popconfirm>
          )}
        </div>
      )}
      <table className="w-full text-sm">
        <thead className="bg-gray-50">
          <tr>
            {onDelete && <th className="px-3 py-2 w-8" />}
            <th className="px-4 py-2 text-left font-medium text-gray-600">#</th>
            <th className="px-4 py-2 text-left font-medium text-gray-600">
              {t("resultTable.category")}
            </th>
            <th className="px-4 py-2 text-left font-medium text-gray-600">x1</th>
            <th className="px-4 py-2 text-left font-medium text-gray-600">y1</th>
            <th className="px-4 py-2 text-left font-medium text-gray-600">x2</th>
            <th className="px-4 py-2 text-left font-medium text-gray-600">y2</th>
            <th className="px-4 py-2 text-left font-medium text-gray-600">
              {t("resultTable.confidence")}
            </th>
            <th className="px-4 py-2 text-left font-medium text-gray-600">Mask</th>
            <th className="px-4 py-2 w-10" />
            {onDelete && <th className="px-4 py-2 w-12" />}
          </tr>
        </thead>
        <tbody>
          {boxes.map((box, i) => (
            <tr key={box.id} className="border-t border-gray-100 hover:bg-gray-50">
              {onDelete && (
                <td className="px-3 py-2">
                  <input
                    type="checkbox"
                    checked={selectedIds.has(box.id)}
                    onChange={() => toggleOne(box.id)}
                    className="h-3.5 w-3.5 rounded border-gray-300"
                  />
                </td>
              )}
              <td className="px-4 py-2 text-gray-400">{i + 1}</td>
              <td className="px-4 py-2 font-medium text-gray-800">{box.className}</td>
              <td className="px-4 py-2 text-gray-600">{box.x1}</td>
              <td className="px-4 py-2 text-gray-600">{box.y1}</td>
              <td className="px-4 py-2 text-gray-600">{box.x2}</td>
              <td className="px-4 py-2 text-gray-600">{box.y2}</td>
              <td className="px-4 py-2 text-gray-600">
                {box.confidence != null ? box.confidence.toFixed(3) : "-"}
              </td>
              <td className="px-4 py-2 text-gray-500">
                {box.maskPolygon && box.maskPolygon.length >= 3
                  ? `${box.maskPolygon.length} pts`
                  : "-"}
              </td>
              <td className="px-2 py-2">
                <button
                  onClick={() => onToggleVisibility(box.id)}
                  className="text-gray-400 hover:text-gray-600 transition-colors"
                  title={
                    hiddenIndices.has(box.id) ? t("resultTable.showBox") : t("resultTable.hideBox")
                  }
                >
                  {hiddenIndices.has(box.id) ? (
                    <svg
                      className="h-4 w-4"
                      fill="none"
                      viewBox="0 0 24 24"
                      stroke="currentColor"
                      strokeWidth={1.5}
                    >
                      <path
                        strokeLinecap="round"
                        strokeLinejoin="round"
                        d="M3.98 8.223A10.477 10.477 0 001.934 12C3.226 16.338 7.244 19.5 12 19.5c.993 0 1.953-.138 2.863-.395M6.228 6.228A10.45 10.45 0 0112 4.5c4.756 0 8.773 3.162 10.065 7.498a10.523 10.523 0 01-4.293 5.774M6.228 6.228L3 3m3.228 3.228l3.65 3.65m7.894 7.894L21 21m-3.228-3.228l-3.65-3.65m0 0a3 3 0 10-4.243-4.243m4.242 4.242L9.88 9.88"
                      />
                    </svg>
                  ) : (
                    <svg
                      className="h-4 w-4"
                      fill="none"
                      viewBox="0 0 24 24"
                      stroke="currentColor"
                      strokeWidth={1.5}
                    >
                      <path
                        strokeLinecap="round"
                        strokeLinejoin="round"
                        d="M2.036 12.322a1.012 1.012 0 010-.639C3.423 7.51 7.36 4.5 12 4.5c4.638 0 8.573 3.007 9.963 7.178.07.207.07.431 0 .639C20.577 16.49 16.64 19.5 12 19.5c-4.638 0-8.573-3.007-9.963-7.178z"
                      />
                      <path
                        strokeLinecap="round"
                        strokeLinejoin="round"
                        d="M15 12a3 3 0 11-6 0 3 3 0 016 0z"
                      />
                    </svg>
                  )}
                </button>
              </td>
              {onDelete && (
                <td className="px-2 py-2">
                  <Popconfirm
                    title={t("resultTable.deleteBoxConfirm")}
                    onConfirm={() => onDelete(box.id)}
                    okText={t("common.delete")}
                    cancelText={t("common.cancel")}
                    okButtonProps={{ danger: true }}
                  >
                    <button
                      disabled={deletingSelected}
                      className="text-xs text-red-400 hover:text-red-600 transition-colors"
                      title={t("resultTable.deleteBox")}
                    >
                      ✕
                    </button>
                  </Popconfirm>
                </td>
              )}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
