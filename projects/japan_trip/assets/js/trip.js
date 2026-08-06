(function () {
  "use strict";

  var data = window.JAPAN_TRIP_DATA;
  var mapElement = document.getElementById("trip-map");
  var mapShell = document.querySelector(".map-shell");
  var mapExpandButton = document.querySelector(".map-expand");
  var mapTitle = document.getElementById("map-title");
  var mapStatus = document.getElementById("map-status");
  var mapError = document.querySelector(".map-error");
  var dayTabs = Array.from(document.querySelectorAll(".day-tab"));
  var daySections = Array.from(document.querySelectorAll("[data-day-section]"));
  var selectedDay = "all";
  var map;
  var routeLayers = new Map();
  var markerLayers = new Map();
  var activeStopIds = [];
  var routeLoadPromise = Promise.resolve();

  function escapeHtml(value) {
    return String(value)
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;")
      .replace(/'/g, "&#039;");
  }

  function getDay(dayId) {
    return data.days.find(function (day) {
      return day.id === Number(dayId);
    });
  }

  function getStop(stopId) {
    return data.stops.find(function (stop) {
      return stop.id === stopId;
    });
  }

  function getSelectedColor(stop) {
    if (selectedDay !== "all") {
      return getDay(selectedDay).color;
    }
    if (stop.days.length === 1) {
      return getDay(stop.days[0]).color;
    }
    return "#17231f";
  }

  function markerLabel(stop) {
    if (selectedDay !== "all") {
      var day = getDay(selectedDay);
      var index = day.stops.indexOf(stop.id);
      return index >= 0 ? String(index + 1) : "·";
    }
    return stop.days.length > 1 ? stop.days.join("·") : String(stop.days[0]);
  }

  function createMarkerIcon(stop, isActive) {
    var classes = ["trip-marker"];
    if (stop.optional) {
      classes.push("is-optional");
    }
    if (isActive) {
      classes.push("is-active");
    }

    return window.L.divIcon({
      className: "trip-marker-icon",
      html: "<div class=\"" + classes.join(" ") + "\" style=\"--marker-color:" + getSelectedColor(stop) + "\"><span>" + markerLabel(stop) + "</span></div>",
      iconSize: [31, 31],
      iconAnchor: [15, 29],
      popupAnchor: [0, -28],
    });
  }

  function createPopupContent(stop) {
    var dayText = stop.days.map(function (day) { return "Day " + day; }).join(" · ");
    var mapsUrl = "https://www.google.com/maps/search/?api=1&query=" + encodeURIComponent(stop.query);
    return "<div class=\"trip-popup\"><h3>" + escapeHtml(stop.name) + "</h3><p>" + escapeHtml(dayText + " · " + stop.label) + "</p><a href=\"" + mapsUrl + "\" target=\"_blank\" rel=\"noreferrer\">Google 지도에서 보기 ↗</a></div>";
  }

  function setActiveMarkers(stopIds) {
    activeStopIds = stopIds.slice();
    markerLayers.forEach(function (marker, stopId) {
      var stop = getStop(stopId);
      marker.setIcon(createMarkerIcon(stop, activeStopIds.includes(stopId)));
    });
  }

  function highlightCards(stopIds) {
    document.querySelectorAll(".timeline-card.is-active").forEach(function (card) {
      card.classList.remove("is-active");
    });

    var firstCard;
    document.querySelectorAll(".timeline-card[data-stop-ids]").forEach(function (card) {
      var cardStopIds = card.dataset.stopIds.split(/\s+/);
      if (stopIds.some(function (id) { return cardStopIds.includes(id); })) {
        card.classList.add("is-active");
        firstCard = firstCard || card;
      }
    });
    return firstCard;
  }

  function focusStops(stopIds, options) {
    if (!map) {
      return;
    }

    var settings = options || {};
    var validStops = stopIds.map(getStop).filter(Boolean);
    if (!validStops.length) {
      return;
    }

    setActiveMarkers(validStops.map(function (stop) { return stop.id; }));
    highlightCards(validStops.map(function (stop) { return stop.id; }));

    if (validStops.length === 1) {
      map.flyTo([validStops[0].lat, validStops[0].lng], Math.max(map.getZoom(), 13), { duration: 0.65 });
      var marker = markerLayers.get(validStops[0].id);
      if (marker) {
        window.setTimeout(function () { marker.openPopup(); }, 220);
      }
    } else {
      var bounds = window.L.latLngBounds(validStops.map(function (stop) { return [stop.lat, stop.lng]; }));
      map.flyToBounds(bounds, { padding: [44, 44], maxZoom: 12, duration: 0.65 });
    }

    mapStatus.textContent = validStops.map(function (stop) { return stop.name; }).join(", ") + " 위치를 지도에 표시했습니다.";

    if (settings.expandOnMobile && window.matchMedia("(max-width: 820px)").matches) {
      setMapExpanded(true);
    }
  }

  function createSchematicLayer(day) {
    var points = day.stops.map(getStop).filter(Boolean).map(function (stop) {
      return [stop.lat, stop.lng];
    });
    return window.L.polyline(points, {
      color: day.color,
      weight: 4,
      opacity: 0.9,
      dashArray: "7 9",
      lineCap: "round",
      lineJoin: "round",
    });
  }

  function createGeoJsonLayer(day, feature) {
    return window.L.geoJSON(feature, {
      style: {
        color: day.color,
        weight: 4,
        opacity: 0.88,
        lineCap: "round",
        lineJoin: "round",
      },
    });
  }

  function loadRoutes() {
    var requests = data.days.map(function (day) {
      if (day.routeType === "schematic") {
        routeLayers.set(day.id, createSchematicLayer(day));
        return Promise.resolve();
      }

      return fetch(day.routeFile)
        .then(function (response) {
          if (!response.ok) {
            throw new Error("Route response " + response.status);
          }
          return response.json();
        })
        .then(function (feature) {
          routeLayers.set(day.id, createGeoJsonLayer(day, feature));
        })
        .catch(function () {
          routeLayers.set(day.id, createSchematicLayer(day));
        });
    });

    return Promise.all(requests).then(function () {
      applyMapFilter();
      fitMapToSelection(false);
    });
  }

  function applyMapFilter() {
    if (!map) {
      return;
    }

    routeLayers.forEach(function (layer, dayId) {
      var shouldShow = selectedDay === "all" || Number(selectedDay) === dayId;
      if (shouldShow && !map.hasLayer(layer)) {
        layer.addTo(map);
      } else if (!shouldShow && map.hasLayer(layer)) {
        map.removeLayer(layer);
      }
    });

    markerLayers.forEach(function (marker, stopId) {
      var stop = getStop(stopId);
      var shouldShow = selectedDay === "all" || stop.days.includes(Number(selectedDay));
      if (shouldShow && !map.hasLayer(marker)) {
        marker.addTo(map);
      } else if (!shouldShow && map.hasLayer(marker)) {
        map.removeLayer(marker);
      }
      marker.setIcon(createMarkerIcon(stop, activeStopIds.includes(stopId)));
    });
  }

  function selectionBounds() {
    var bounds = window.L.latLngBounds([]);
    var daysToFit = selectedDay === "all" ? data.days : [getDay(selectedDay)];

    daysToFit.forEach(function (day) {
      var layer = routeLayers.get(day.id);
      if (layer && typeof layer.getBounds === "function") {
        bounds.extend(layer.getBounds());
      }
      day.stops.map(getStop).filter(Boolean).forEach(function (stop) {
        bounds.extend([stop.lat, stop.lng]);
      });
    });
    return bounds;
  }

  function fitMapToSelection(animate) {
    if (!map) {
      return;
    }
    var bounds = selectionBounds();
    if (bounds.isValid()) {
      map.fitBounds(bounds, {
        animate: Boolean(animate),
        duration: 0.55,
        padding: window.matchMedia("(max-width: 520px)").matches ? [24, 24] : [42, 42],
        maxZoom: selectedDay === "all" ? 9 : 11,
      });
    }
  }

  function selectDay(dayValue, options) {
    var settings = options || {};
    selectedDay = String(dayValue);
    activeStopIds = [];

    dayTabs.forEach(function (tab) {
      var isActive = tab.dataset.day === selectedDay;
      tab.classList.toggle("is-active", isActive);
      tab.setAttribute("aria-pressed", String(isActive));
    });

    daySections.forEach(function (section) {
      section.hidden = selectedDay !== "all" && section.dataset.daySection !== selectedDay;
    });

    document.querySelectorAll(".timeline-card.is-active").forEach(function (card) {
      card.classList.remove("is-active");
    });

    if (selectedDay === "all") {
      mapTitle.textContent = "전체 이동 동선";
      mapStatus.textContent = "전체 여행 경로를 표시했습니다.";
    } else {
      var day = getDay(selectedDay);
      mapTitle.textContent = "Day " + day.id + " · " + day.title;
      mapStatus.textContent = "Day " + day.id + " 경로를 표시했습니다.";
    }

    applyMapFilter();
    routeLoadPromise.then(function () {
      fitMapToSelection(true);
    });

    if (settings.scrollToAtlas) {
      document.querySelector(".atlas-heading").scrollIntoView({ behavior: "smooth", block: "start" });
    }
  }

  function setMapExpanded(expanded) {
    if (!mapShell || !mapExpandButton) {
      return;
    }
    mapShell.classList.toggle("is-expanded", expanded);
    document.body.classList.toggle("map-open", expanded);
    mapExpandButton.setAttribute("aria-expanded", String(expanded));
    mapExpandButton.querySelector(".expand-label").textContent = expanded ? "닫기" : "크게 보기";
    mapExpandButton.lastElementChild.textContent = expanded ? "×" : "↗";

    if (expanded) {
      mapShell.setAttribute("role", "dialog");
      mapShell.setAttribute("aria-modal", "true");
      mapShell.setAttribute("aria-label", "확대된 여행 경로 지도");
    } else {
      mapShell.removeAttribute("role");
      mapShell.removeAttribute("aria-modal");
      mapShell.removeAttribute("aria-label");
    }

    if (map) {
      window.setTimeout(function () {
        map.invalidateSize();
        if (activeStopIds.length) {
          focusStops(activeStopIds);
        } else {
          fitMapToSelection(false);
        }
      }, 120);
    }
  }

  function initializeMap() {
    if (!data || !mapElement || !window.L) {
      if (mapError) {
        mapError.hidden = false;
      }
      if (mapElement) {
        mapElement.querySelector(".map-loading").textContent = "지도를 불러오지 못했습니다.";
      }
      return;
    }

    map = window.L.map(mapElement, {
      center: [33.16, 131.06],
      zoom: 8,
      scrollWheelZoom: false,
      zoomControl: true,
      preferCanvas: true,
    });

    window.L.tileLayer("https://tile.openstreetmap.org/{z}/{x}/{y}.png", {
      maxZoom: 19,
      attribution: "&copy; <a href=\"https://www.openstreetmap.org/copyright\">OpenStreetMap</a> contributors",
    }).addTo(map);

    data.stops.forEach(function (stop) {
      var marker = window.L.marker([stop.lat, stop.lng], {
        icon: createMarkerIcon(stop, false),
        title: stop.name,
        riseOnHover: true,
      });
      marker.bindPopup(createPopupContent(stop), { maxWidth: 250, minWidth: 160 });
      marker.on("click", function () {
        setActiveMarkers([stop.id]);
        highlightCards([stop.id]);
        mapStatus.textContent = stop.name + " 위치를 선택했습니다.";
      });
      markerLayers.set(stop.id, marker);
    });

    applyMapFilter();
    routeLoadPromise = loadRoutes();
  }

  function initializeControls() {
    dayTabs.forEach(function (tab) {
      tab.addEventListener("click", function () {
        selectDay(tab.dataset.day);
      });
    });

    document.querySelectorAll("[data-day-select]").forEach(function (button) {
      button.addEventListener("click", function () {
        selectDay(button.dataset.daySelect, { scrollToAtlas: true });
      });
    });

    document.querySelectorAll(".map-focus").forEach(function (button) {
      button.addEventListener("click", function () {
        var stopIds = button.dataset.stopIds.split(/\s+/);
        if (window.matchMedia("(max-width: 820px)").matches) {
          setMapExpanded(true);
          window.setTimeout(function () { focusStops(stopIds); }, 150);
        } else {
          focusStops(stopIds);
        }
      });
    });

    if (mapExpandButton) {
      mapExpandButton.addEventListener("click", function () {
        setMapExpanded(mapExpandButton.getAttribute("aria-expanded") !== "true");
      });
    }

    document.addEventListener("keydown", function (event) {
      if (event.key === "Escape" && mapShell && mapShell.classList.contains("is-expanded")) {
        setMapExpanded(false);
        mapExpandButton.focus();
      }
    });
  }

  function initializeChecklist() {
    var storageKey = "japan-trip-checklist-v1";
    var checkboxes = Array.from(document.querySelectorAll("[data-check]"));
    var saved = {};

    try {
      saved = JSON.parse(window.localStorage.getItem(storageKey) || "{}");
    } catch (error) {
      saved = {};
    }

    checkboxes.forEach(function (checkbox) {
      checkbox.checked = Boolean(saved[checkbox.dataset.check]);
      checkbox.addEventListener("change", function () {
        saved[checkbox.dataset.check] = checkbox.checked;
        try {
          window.localStorage.setItem(storageKey, JSON.stringify(saved));
        } catch (error) {
          // The checklist remains usable when storage is unavailable.
        }
      });
    });

    var resetButton = document.querySelector(".checklist-reset");
    if (resetButton) {
      resetButton.addEventListener("click", function () {
        saved = {};
        checkboxes.forEach(function (checkbox) { checkbox.checked = false; });
        try {
          window.localStorage.removeItem(storageKey);
        } catch (error) {
          // Nothing else is required when storage is unavailable.
        }
      });
    }
  }

  function dateInTimezone(timezone) {
    var parts = new Intl.DateTimeFormat("en", {
      timeZone: timezone,
      year: "numeric",
      month: "2-digit",
      day: "2-digit",
    }).formatToParts(new Date());
    var values = {};
    parts.forEach(function (part) { values[part.type] = part.value; });
    return values.year + "-" + values.month + "-" + values.day;
  }

  function updateTripStatus() {
    var statusElement = document.getElementById("trip-status");
    if (!statusElement) {
      return;
    }

    var today = dateInTimezone(data.trip.timezone);
    var todayDate = new Date(today + "T00:00:00Z");
    var startDate = new Date(data.trip.start + "T00:00:00Z");
    var endDate = new Date(data.trip.end + "T00:00:00Z");
    var dayMilliseconds = 24 * 60 * 60 * 1000;

    if (todayDate < startDate) {
      statusElement.textContent = "출발까지 D-" + Math.round((startDate - todayDate) / dayMilliseconds);
    } else if (todayDate <= endDate) {
      var tripDay = Math.floor((todayDate - startDate) / dayMilliseconds) + 1;
      statusElement.textContent = "여행 중 · DAY " + tripDay;
      window.setTimeout(function () { selectDay(String(tripDay)); }, 0);
    } else {
      statusElement.textContent = "JOURNEY COMPLETE";
    }
  }

  function initializeReveal() {
    var revealElements = document.querySelectorAll(".reveal");
    if (!("IntersectionObserver" in window) || window.matchMedia("(prefers-reduced-motion: reduce)").matches) {
      revealElements.forEach(function (element) { element.classList.add("is-visible"); });
      return;
    }

    var observer = new IntersectionObserver(function (entries) {
      entries.forEach(function (entry) {
        if (entry.isIntersecting) {
          entry.target.classList.add("is-visible");
          observer.unobserve(entry.target);
        }
      });
    }, { rootMargin: "0px 0px -6%", threshold: 0.08 });

    revealElements.forEach(function (element) { observer.observe(element); });
  }

  var appInitialized = false;

  function initializeTripApp() {
    if (appInitialized) {
      return;
    }

    appInitialized = true;
    initializeControls();
    initializeChecklist();
    updateTripStatus();
    initializeReveal();
    initializeMap();
  }

  document.addEventListener("trip:unlocked", initializeTripApp);
  document.addEventListener("trip:locked", function () {
    if (mapShell && mapShell.classList.contains("is-expanded")) {
      setMapExpanded(false);
    }
  });

  if (!document.body.classList.contains("trip-locked")) {
    initializeTripApp();
  }
}());
