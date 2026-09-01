import type { ButtonHTMLAttributes } from "react";

import { cn } from "@/lib";

type Variant = "primary" | "ghost" | "outline";

const VARIANTS: Record<Variant, string> = {
  primary:
    "bg-primary text-primary-foreground hover:bg-primary-hover active:scale-[0.98] disabled:opacity-60",
  ghost: "text-muted-foreground hover:text-primary",
  outline:
    "border border-border text-muted-foreground hover:border-accent-soft hover:text-primary",
};

interface ButtonProps extends ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: Variant;
}

export function Button({ variant = "primary", className, type = "button", ...props }: ButtonProps) {
  return (
    <button
      type={type}
      className={cn(
        "rounded-xl px-4 py-2.5 text-sm font-semibold transition-all disabled:cursor-not-allowed",
        VARIANTS[variant],
        className,
      )}
      {...props}
    />
  );
}

export default Button;
