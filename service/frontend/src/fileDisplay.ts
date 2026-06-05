export function fileDisplayText(file: File | null, label: string): string {
  return file?.name ?? `${label} 업로드 대기`;
}
