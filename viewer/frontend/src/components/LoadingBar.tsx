type LoadingBarProps = {
  active: boolean;
  label?: string;
};

export function LoadingBar({ active, label = "Loading" }: LoadingBarProps) {
  return (
    <div
      className={`loading-strip${active ? " active" : ""}`}
      aria-hidden={!active}
      aria-label={label}
    >
      <div className="loading-strip-bar" />
    </div>
  );
}
