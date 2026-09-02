import { Link, useLocation } from "react-router";

import { paths } from "@/lib";
import { Icon, NavBar } from "@/components/ui";

export function NotFoundPage() {
  const { pathname } = useLocation();

  return (
    <div className="flex h-full flex-col bg-background">
      <NavBar homeTo={paths.home} />
      <div className="flex flex-1 items-center justify-center px-6 py-16">
        <div className="flex max-w-md flex-col items-center gap-4 text-center">
          <div className="flex h-12 w-12 items-center justify-center rounded-2xl bg-muted text-primary">
            <Icon name="alert" size={22} />
          </div>
          <h1 className="text-2xl text-foreground">Page not found</h1>
          <p className="text-sm text-muted-foreground">
            Nothing lives at <span className="break-all font-mono text-xs">{pathname}</span>. A
            study opens at <span className="font-mono text-xs">/&#123;StudyInstanceUID&#125;</span>.
          </p>
          <Link
            to={paths.home}
            className="flex items-center gap-1.5 rounded-full border border-border px-3 py-1.5 text-xs font-medium text-muted-foreground transition-colors hover:border-accent-soft hover:text-primary"
          >
            <Icon name="arrow-left" size={13} />
            Back to studies
          </Link>
        </div>
      </div>
    </div>
  );
}

export default NotFoundPage;
