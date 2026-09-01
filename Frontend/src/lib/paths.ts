/**
 * Every URL the app can be at, in one place.
 *
 * The route patterns live in `@/router`; these are the builders that produce
 * the matching hrefs, so a link and the route it points at cannot drift apart.
 * A StudyInstanceUID is a dotted numeric and needs no escaping in practice, but
 * it arrives from the API rather than from us, hence the encode.
 */
export const paths = {
  home: "/",
  benchmark: "/benchmark",
  processing: "/processing",
  study: (studyUid: string) => `/${encodeURIComponent(studyUid)}`,
} as const;
