/** Loading, empty, and error states — the three things every panel needs. */

import { cn } from "@/lib";
import Icon from "./Icon";

export function Spinner({ className }: { className?: string }) {
  return <Icon name="spinner" size={16} className={cn("animate-spin", className)} />;
}

export function Loading({ label = "Loading…" }: { label?: string }) {
  return (
    <div className="flex items-center justify-center gap-2 py-12 text-sm text-muted-foreground">
      <Spinner />
      {label}
    </div>
  );
}

export function EmptyState({ message }: { message: string }) {
  return <div className="py-12 text-center text-sm text-muted-foreground">{message}</div>;
}

interface ErrorStateProps {
  message: string;
  onRetry?: () => void;
}

export function ErrorState({ message, onRetry }: ErrorStateProps) {
  return (
    <div className="flex flex-col items-center gap-3 rounded-2xl border border-red-100 bg-red-50 px-6 py-8 text-center">
      <Icon name="alert" size={22} className="text-red-500" />
      <p className="text-sm text-red-700">{message}</p>
      {onRetry && (
        <button
          type="button"
          onClick={onRetry}
          className="rounded-full border border-red-200 px-3 py-1 text-xs font-semibold text-red-700 transition-colors hover:bg-red-100"
        >
          Try again
        </button>
      )}
    </div>
  );
}
