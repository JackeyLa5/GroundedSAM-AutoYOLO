export const batchFileMap = new Map<string, File>();
export const batchFileDetectionMap = new Map<string, string>();

export function getFileIdentity(file: File): string {
  return `${file.name}:${file.size}:${file.lastModified}`;
}

export const fileUrlCache = new WeakMap<File, string>();

export function getFileUrl(file: File): string {
  if (!fileUrlCache.has(file)) {
    fileUrlCache.set(file, URL.createObjectURL(file));
  }
  return fileUrlCache.get(file)!;
}
