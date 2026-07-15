import type { TrainingJob } from "@/types";

export function displayModelVariant(modelVariant: string, taskType?: string | null): string {
  if (taskType === "segment" && !modelVariant.endsWith("-seg")) {
    return `${modelVariant}-seg`;
  }
  return modelVariant;
}

export function displayTrainingJobModel(job: Pick<TrainingJob, "modelVariant" | "taskType">): string {
  return displayModelVariant(job.modelVariant, job.taskType);
}
