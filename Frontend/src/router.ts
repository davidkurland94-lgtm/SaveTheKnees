import { createBrowserRouter } from "react-router";

import App from "@/App";
import { paths } from "@/lib";
import BenchmarkPage from "@/components/pages/BenchmarkPage";
import HomePage from "@/components/pages/HomePage";
import NotFoundPage from "@/components/pages/NotFoundPage";
import ProcessingPage from "@/components/pages/ProcessingPage";
import StudyPage from "@/components/pages/StudyPage";

export const router = createBrowserRouter([
  {
    path: paths.home,
    Component: App,
    children: [
      { index: true, Component: HomePage },
      { path: "benchmark", Component: BenchmarkPage },
      { path: "processing", Component: ProcessingPage },
      { path: ":studyUid", Component: StudyPage },
      { path: "*", Component: NotFoundPage },
    ],
  },
]);
