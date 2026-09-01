import type { ReactNode } from "react";

import Icon from "./Icon";

interface NavBarProps {
  onHome?: () => void;
  children?: ReactNode;
}

export function NavBar({ onHome, children }: NavBarProps) {
  return (
    <header className="sticky top-0 z-10 flex items-center justify-between border-b border-border bg-white px-6 py-4">
      <button
        type="button"
        onClick={onHome}
        disabled={!onHome}
        className="flex items-center gap-3 text-left enabled:cursor-pointer"
      >
        <div className="flex h-8 w-8 items-center justify-center rounded-lg bg-primary text-white">
          <Icon name="logo" size={18} strokeWidth={2} />
        </div>
        <div>
          <div className="text-sm font-semibold text-foreground">Save the Knees</div>
          <div className="text-xs text-muted-foreground">Knee MRI finding detection</div>
        </div>
      </button>
      <div className="flex items-center gap-3">{children}</div>
    </header>
  );
}

export default NavBar;
