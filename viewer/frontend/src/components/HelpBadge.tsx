import type { ReactNode } from "react";

type HelpBadgeProps = {
  text?: string;
  children?: ReactNode;
};

export function HelpBadge({ text, children }: HelpBadgeProps) {
  const label = text ?? (typeof children === "string" ? children : "Help");
  return (
    <div className="help-badge" aria-label={label}>
      <span>i</span>
      <div className="help-popover">{children ?? text}</div>
    </div>
  );
}
