import { useCallback, useState } from "react";

/**
 * One piece of state that works controlled or not.
 *
 * Pass `controlled` to drive it from the parent (e.g. to keep the 2D and 3D
 * viewers on the same series); leave it `undefined` and the component owns the
 * value while `onChange` still reports every move.
 */
export function useControlled<T>(
  controlled: T | undefined,
  fallback: T,
  onChange?: (value: T) => void,
): [T, (value: T) => void] {
  const [internal, setInternal] = useState(fallback);
  const isControlled = controlled !== undefined;

  const set = useCallback(
    (value: T) => {
      if (!isControlled) setInternal(value);
      onChange?.(value);
    },
    [isControlled, onChange],
  );

  return [isControlled ? controlled : internal, set];
}
