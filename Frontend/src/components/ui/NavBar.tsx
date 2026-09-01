import type { ReactNode } from "react";
import { Link } from "react-router";

import Icon from "./Icon";

interface NavBarProps {
  /**
   * Where the brand links to. Left out it renders as plain text, which is what
   * the home page wants (it is already there) and what the processing screen
   * wants (leaving mid-run would throw the decoded series away).
   */
  homeTo?: string;
  children?: ReactNode;
}

export function NavBar({ homeTo, children }: NavBarProps) {
  const brand = (
    <>
      <div className="flex h-8 w-8 items-center justify-center rounded-lg bg-primary text-white">
        <Icon name="logo" size={18} strokeWidth={2} />
      </div>
      <div>
        <div className="text-sm font-semibold text-foreground">Save the Knees</div>
        <div className="text-xs text-muted-foreground">Knee MRI finding detection</div>
      </div>
    </>
  );

  return (
    <header className="sticky top-0 z-10 flex items-center justify-between border-b border-border bg-white px-6 py-4">
      {homeTo ? (
        <Link to={homeTo} className="flex items-center gap-3 text-left">
          {brand}
        </Link>
      ) : (
        <div className="flex items-center gap-3 text-left">{brand}</div>
      )}
      <div className="flex items-center gap-3">{children}</div>
    </header>
  );
}

export default NavBar;
