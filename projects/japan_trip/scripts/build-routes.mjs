import { mkdir, writeFile } from "node:fs/promises";
import { fileURLToPath } from "node:url";
import path from "node:path";

const scriptDirectory = path.dirname(fileURLToPath(import.meta.url));
const outputDirectory = path.resolve(scriptDirectory, "../assets/routes");

const routes = [
  {
    day: 1,
    stops: [
      [130.4442945, 33.5848221],
      [130.4198799, 33.5902769],
      [130.4223, 33.5886],
      [130.5741717, 33.6181923],
      [130.4098021, 33.6053216],
      [130.406708, 33.603493],
      [130.4060551, 33.5936772],
      [130.40562, 33.59348],
    ],
  },
  {
    day: 2,
    stops: [
      [130.4442945, 33.5848221],
      [131.4114362, 33.3539976],
      [131.4687332, 33.31683],
      [131.4724531, 33.3164456],
      [131.5356401, 33.2589685],
      [131.485325, 33.315669],
      [131.4831814, 33.3196602],
      [131.4793204, 33.3144306],
      [131.496689, 33.272907],
    ],
  },
  {
    day: 3,
    stops: [
      [131.496689, 33.272907],
      [131.0502492, 32.8849747],
      [131.0847112, 32.8841192],
      [131.300939, 32.7017851],
      [131.4511721, 32.9643788],
      [131.496689, 33.272907],
    ],
  },
  {
    day: 4,
    stops: [
      [131.496689, 33.272907],
      [130.4442945, 33.5848221],
    ],
  },
];

function squaredDistance(pointA, pointB) {
  const x = pointA[0] - pointB[0];
  const y = pointA[1] - pointB[1];
  return x * x + y * y;
}

function squaredSegmentDistance(point, start, end) {
  let x = start[0];
  let y = start[1];
  let dx = end[0] - x;
  let dy = end[1] - y;

  if (dx !== 0 || dy !== 0) {
    const ratio = ((point[0] - x) * dx + (point[1] - y) * dy) / (dx * dx + dy * dy);
    if (ratio > 1) {
      x = end[0];
      y = end[1];
    } else if (ratio > 0) {
      x += dx * ratio;
      y += dy * ratio;
    }
  }

  dx = point[0] - x;
  dy = point[1] - y;
  return dx * dx + dy * dy;
}

function simplifyRadialDistance(points, squaredTolerance) {
  let previousPoint = points[0];
  const simplified = [previousPoint];

  for (let index = 1; index < points.length; index += 1) {
    const point = points[index];
    if (squaredDistance(point, previousPoint) > squaredTolerance) {
      simplified.push(point);
      previousPoint = point;
    }
  }

  if (previousPoint !== points.at(-1)) {
    simplified.push(points.at(-1));
  }

  return simplified;
}

function simplifyDouglasPeuckerStep(points, first, last, squaredTolerance, simplified) {
  let maxSquaredDistance = squaredTolerance;
  let splitIndex;

  for (let index = first + 1; index < last; index += 1) {
    const distance = squaredSegmentDistance(points[index], points[first], points[last]);
    if (distance > maxSquaredDistance) {
      splitIndex = index;
      maxSquaredDistance = distance;
    }
  }

  if (maxSquaredDistance > squaredTolerance && splitIndex !== undefined) {
    if (splitIndex - first > 1) {
      simplifyDouglasPeuckerStep(points, first, splitIndex, squaredTolerance, simplified);
    }
    simplified.push(points[splitIndex]);
    if (last - splitIndex > 1) {
      simplifyDouglasPeuckerStep(points, splitIndex, last, squaredTolerance, simplified);
    }
  }
}

function simplifyLine(points, tolerance = 0.00015) {
  if (points.length <= 2) {
    return points;
  }

  const squaredTolerance = tolerance * tolerance;
  const radialPoints = simplifyRadialDistance(points, squaredTolerance);
  const lastIndex = radialPoints.length - 1;
  const simplified = [radialPoints[0]];
  simplifyDouglasPeuckerStep(radialPoints, 0, lastIndex, squaredTolerance, simplified);
  simplified.push(radialPoints[lastIndex]);
  return simplified;
}

async function fetchRoute(route) {
  const coordinates = route.stops.map((stop) => stop.join(",")).join(";");
  const url = "https://router.project-osrm.org/route/v1/driving/" + coordinates
    + "?overview=full&geometries=geojson&steps=false";
  const response = await fetch(url, {
    headers: {
      "User-Agent": "OnJungWoanJapanTripPage/1.0 (+https://on-jungwoan.github.io)",
    },
  });

  if (!response.ok) {
    throw new Error("Day " + route.day + " route request failed: " + response.status);
  }

  const result = await response.json();
  const firstRoute = result.routes?.[0];
  if (!firstRoute) {
    throw new Error("Day " + route.day + " returned no route");
  }

  const originalCoordinates = firstRoute.geometry.coordinates;
  const simplifiedCoordinates = simplifyLine(originalCoordinates);

  return {
    type: "Feature",
    properties: {
      day: route.day,
      source: "OSRM",
      distance_km: Math.round(firstRoute.distance / 100) / 10,
      duration_minutes: Math.round(firstRoute.duration / 60),
      original_points: originalCoordinates.length,
      simplified_points: simplifiedCoordinates.length,
    },
    geometry: {
      type: "LineString",
      coordinates: simplifiedCoordinates,
    },
  };
}

await mkdir(outputDirectory, { recursive: true });

for (const route of routes) {
  const feature = await fetchRoute(route);
  const outputPath = path.join(outputDirectory, "day-" + route.day + ".geojson");
  await writeFile(outputPath, JSON.stringify(feature), "utf8");
  console.log("Wrote " + path.relative(process.cwd(), outputPath));
}
