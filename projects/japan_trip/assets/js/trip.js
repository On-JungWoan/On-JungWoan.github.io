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
  var liveEvents = data && Array.isArray(data.liveEvents) ? data.liveEvents : [];
  var walletItems = data && Array.isArray(data.walletItems) ? data.walletItems : [];
  var walletDialog = document.getElementById("travel-wallet");
  var walletOpenButton = document.querySelector(".wallet-open");
  var walletCloseButton = document.querySelector(".wallet-close");
  var walletItemsElement = document.getElementById("wallet-items");
  var walletStatus = document.getElementById("wallet-status");
  var liveDay = document.getElementById("live-day");
  var liveKicker = document.getElementById("live-kicker");
  var liveTitle = document.getElementById("live-title");
  var liveDestination = document.getElementById("live-destination");
  var liveCountdownLabel = document.getElementById("live-countdown-label");
  var liveCountdown = document.getElementById("live-countdown");
  var liveDirections = document.getElementById("live-directions");
  var liveStatus = document.getElementById("live-status");
  var selectedDay = "all";
  var manualDaySelection = false;
  var lastLiveEventId = "";
  var lastWalletOpener;
  var liveTimer;
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
    if (settings.userInitiated) {
      manualDaySelection = true;
    }
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
        selectDay(tab.dataset.day, { userInitiated: true });
      });
    });

    document.querySelectorAll("[data-day-select]").forEach(function (button) {
      button.addEventListener("click", function () {
        selectDay(button.dataset.daySelect, { scrollToAtlas: true, userInitiated: true });
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

  function setMapLinkLabel(link, label) {
    link.textContent = label + " ";
    var arrow = document.createElement("span");
    arrow.setAttribute("aria-hidden", "true");
    arrow.textContent = "↗";
    link.appendChild(arrow);
  }

  function createDirectionsUrl(directions) {
    if (!directions) {
      return "";
    }

    var origin = getStop(directions.origin);
    var destination = getStop(directions.destination);
    if (!origin || !destination) {
      return "";
    }

    var params = new URLSearchParams();
    params.set("api", "1");
    params.set("origin", origin.query);
    params.set("destination", destination.query);
    params.set("travelmode", directions.mode || "driving");
    params.set("dir_action", "navigate");
    return "https://www.google.com/maps/dir/?" + params.toString();
  }

  function initializeDirections() {
    document.querySelectorAll(".timeline-card .external-map").forEach(function (link) {
      setMapLinkLabel(link, "장소 보기");
    });

    liveEvents.forEach(function (tripEvent) {
      var card = document.querySelector("[data-event-id=\"" + tripEvent.id + "\"]");
      var link = card && card.querySelector(".external-map");
      var directionsUrl = createDirectionsUrl(tripEvent.directions);
      if (!link || !directionsUrl) {
        return;
      }

      link.href = directionsUrl;
      link.classList.add("is-directions");
      link.setAttribute("aria-label", tripEvent.destination + " 길찾기 시작");
      setMapLinkLabel(link, "길찾기 시작");
    });
  }

  function dateInTimezone(timezone, dateValue) {
    var parts = new Intl.DateTimeFormat("en", {
      timeZone: timezone,
      year: "numeric",
      month: "2-digit",
      day: "2-digit",
    }).formatToParts(dateValue || new Date());
    var values = {};
    parts.forEach(function (part) { values[part.type] = part.value; });
    return values.year + "-" + values.month + "-" + values.day;
  }

  function tripDayFromDate(dateString) {
    var date = new Date(dateString + "T00:00:00Z");
    var start = new Date(data.trip.start + "T00:00:00Z");
    var day = Math.floor((date - start) / (24 * 60 * 60 * 1000)) + 1;
    return day >= 1 && day <= data.days.length ? day : 0;
  }

  function calendarDaysBetween(fromDate, toDate) {
    var from = new Date(fromDate + "T00:00:00Z");
    var to = new Date(toDate + "T00:00:00Z");
    return Math.max(0, Math.round((to - from) / (24 * 60 * 60 * 1000)));
  }

  function formatCountdown(milliseconds) {
    var minutes = Math.max(0, Math.ceil(milliseconds / (60 * 1000)));
    if (minutes < 1) {
      return "곧 출발";
    }
    if (minutes < 60) {
      return minutes + "분";
    }

    var hours = Math.floor(minutes / 60);
    var remainingMinutes = minutes % 60;
    return hours + "시간" + (remainingMinutes ? " " + remainingMinutes + "분" : "");
  }

  function getEventStart(tripEvent) {
    var start = new Date(tripEvent.start).getTime();
    var departure = tripEvent.departAt ? new Date(tripEvent.departAt).getTime() : start;
    return Math.min(start, departure);
  }

  function findActiveEvent(nowTime) {
    var active;
    liveEvents.forEach(function (tripEvent) {
      if (nowTime >= getEventStart(tripEvent) && nowTime < new Date(tripEvent.end).getTime()) {
        active = tripEvent;
      }
    });
    return active;
  }

  function findNextDeparture(nowTime) {
    return liveEvents.find(function (tripEvent) {
      return tripEvent.departAt && new Date(tripEvent.departAt).getTime() > nowTime;
    });
  }

  function updateTimelineStates(nowTime, activeEvent, nextEvent) {
    liveEvents.forEach(function (tripEvent) {
      var card = document.querySelector("[data-event-id=\"" + tripEvent.id + "\"]");
      if (!card) {
        return;
      }

      card.classList.remove("is-past", "is-current", "is-next");
      if (nowTime >= new Date(tripEvent.end).getTime()) {
        card.classList.add("is-past");
      }
      if (activeEvent && activeEvent.id === tripEvent.id) {
        card.classList.add("is-current");
      }
      if (nextEvent && nextEvent.id === tripEvent.id) {
        card.classList.add("is-next");
      }
    });
  }

  function setLiveDirections(tripEvent) {
    if (!liveDirections) {
      return;
    }

    var directionsUrl = tripEvent && createDirectionsUrl(tripEvent.directions);
    if (!directionsUrl) {
      liveDirections.hidden = true;
      liveDirections.removeAttribute("href");
      return;
    }

    liveDirections.href = directionsUrl;
    liveDirections.hidden = false;
    liveDirections.setAttribute("aria-label", tripEvent.destination + " 길찾기 시작");
  }

  function renderLiveBar(settings) {
    if (!liveDay || !liveTitle || !liveDestination || !liveCountdown) {
      return;
    }

    liveDay.textContent = settings.day;
    liveKicker.textContent = settings.kicker;
    liveTitle.textContent = settings.title;
    liveDestination.textContent = settings.destination;
    liveCountdownLabel.textContent = settings.countdownLabel;
    liveCountdown.textContent = settings.countdown;
    setLiveDirections(settings.event);

    if (liveStatus && settings.announcementKey !== lastLiveEventId) {
      liveStatus.textContent = settings.announcement;
      lastLiveEventId = settings.announcementKey;
    }
  }

  function updateTripStatus(nowValue) {
    var statusElement = document.getElementById("trip-status");
    if (!statusElement || !data) {
      return;
    }

    var now = nowValue || new Date();
    var today = dateInTimezone(data.trip.timezone, now);
    var todayDate = new Date(today + "T00:00:00Z");
    var startDate = new Date(data.trip.start + "T00:00:00Z");
    var endDate = new Date(data.trip.end + "T00:00:00Z");
    var dayMilliseconds = 24 * 60 * 60 * 1000;

    if (todayDate < startDate) {
      statusElement.textContent = "출발까지 D-" + Math.round((startDate - todayDate) / dayMilliseconds);
    } else if (todayDate <= endDate) {
      var tripDay = Math.floor((todayDate - startDate) / dayMilliseconds) + 1;
      statusElement.textContent = "여행 중 · DAY " + tripDay;
    } else {
      statusElement.textContent = "JOURNEY COMPLETE";
    }
  }

  function updateJourneyNow(nowValue) {
    if (!data || !liveEvents.length) {
      return;
    }

    var now = nowValue || new Date();
    var nowTime = now.getTime();
    var today = dateInTimezone(data.trip.timezone, now);
    var tripDay = tripDayFromDate(today);
    var firstEvent = liveEvents[0];
    var lastEvent = liveEvents[liveEvents.length - 1];
    var activeEvent = findActiveEvent(nowTime);
    var nextEvent = findNextDeparture(nowTime);

    updateTripStatus(now);
    updateTimelineStates(nowTime, activeEvent, nextEvent);

    if (tripDay && !manualDaySelection && selectedDay !== String(tripDay)) {
      selectDay(String(tripDay));
    }

    if (today < data.trip.start) {
      var daysUntilDeparture = calendarDaysBetween(today, data.trip.start);
      renderLiveBar({
        day: "PRE-TRIP",
        kicker: "FIRST DEPARTURE",
        title: firstEvent.title,
        destination: "목적지 · " + firstEvent.destination,
        countdownLabel: "여행까지",
        countdown: "D-" + daysUntilDeparture,
        event: firstEvent,
        announcementKey: "before-" + firstEvent.id,
        announcement: "여행까지 D-" + daysUntilDeparture + ". 첫 일정은 " + firstEvent.title + "입니다.",
      });
      return;
    }

    if (nowTime >= new Date(lastEvent.end).getTime()) {
      renderLiveBar({
        day: "COMPLETE",
        kicker: "JOURNEY ARCHIVE",
        title: "모든 여행 일정을 마쳤습니다.",
        destination: "후쿠오카 · 벳푸 · 아소 · 다카치호",
        countdownLabel: "여행 상태",
        countdown: "완료",
        event: null,
        announcementKey: "complete",
        announcement: "모든 여행 일정을 마쳤습니다.",
      });
      return;
    }

    if (nextEvent) {
      renderLiveBar({
        day: tripDay ? "DAY " + String(tripDay).padStart(2, "0") : "JOURNEY",
        kicker: activeEvent ? "NEXT DEPARTURE" : "UP NEXT",
        title: nextEvent.title,
        destination: "목적지 · " + nextEvent.destination,
        countdownLabel: "출발까지",
        countdown: formatCountdown(new Date(nextEvent.departAt).getTime() - nowTime),
        event: nextEvent,
        announcementKey: "next-" + nextEvent.id,
        announcement: "다음 일정은 " + nextEvent.title + ", 목적지는 " + nextEvent.destination + "입니다.",
      });
      return;
    }

    renderLiveBar({
      day: tripDay ? "DAY " + String(tripDay).padStart(2, "0") : "JOURNEY",
      kicker: "NOW",
      title: activeEvent ? activeEvent.title : "오늘 일정을 마쳤습니다.",
      destination: activeEvent ? "목적지 · " + activeEvent.destination : "다음 날 일정을 확인하세요.",
      countdownLabel: "여행 상태",
      countdown: activeEvent ? "이동 중" : "완료",
      event: null,
      announcementKey: activeEvent ? "active-" + activeEvent.id : "day-complete-" + tripDay,
      announcement: activeEvent ? activeEvent.title + " 일정이 진행 중입니다." : "오늘 일정을 마쳤습니다.",
    });
  }

  function renderWallet() {
    if (!walletItemsElement) {
      return;
    }

    walletItemsElement.innerHTML = walletItems.map(function (item, itemIndex) {
      var fields = item.fields.map(function (field) {
        var hasValue = String(field.value || "").trim().length > 0;
        var href = String(field.href || "").trim();
        var hasSafeLink = hasValue && /^(https?:\/\/|\/(?!\/))/i.test(href);
        var value = hasValue ? escapeHtml(field.value) : "<span class=\"wallet-empty\">미입력</span>";
        if (hasSafeLink) {
          value = "<a class=\"wallet-field-link\" href=\"" + escapeHtml(href) + "\" target=\"_blank\" rel=\"noreferrer\">" + escapeHtml(field.linkLabel || field.value) + "</a>";
        }
        var copyButton = "";
        if (field.copyable) {
          copyButton = "<button type=\"button\" class=\"wallet-copy\" data-copy-value=\"" + escapeHtml(field.value || "") + "\" data-copy-label=\"" + escapeHtml(field.label) + "\"" + (hasValue ? "" : " disabled") + ">복사</button>";
        }
        return "<div class=\"wallet-field\"><dt>" + escapeHtml(field.label) + "</dt><dd><span class=\"wallet-value\">" + value + "</span>" + copyButton + "</dd></div>";
      }).join("");

      return "<details class=\"wallet-ticket wallet-ticket-" + escapeHtml(item.type) + "\"" + (itemIndex === 0 ? " open" : "") + ">" +
        "<summary><span><small>" + escapeHtml(item.kicker) + "</small><strong>" + escapeHtml(item.title) + "</strong><span>" + escapeHtml(item.summary) + "</span></span><i aria-hidden=\"true\">+</i></summary>" +
        "<dl class=\"wallet-fields\">" + fields + "</dl></details>";
    }).join("");
  }

  function fallbackCopy(value) {
    return new Promise(function (resolve, reject) {
      var textarea = document.createElement("textarea");
      textarea.value = value;
      textarea.setAttribute("readonly", "");
      textarea.style.position = "fixed";
      textarea.style.opacity = "0";
      textarea.style.pointerEvents = "none";
      document.body.appendChild(textarea);
      textarea.select();
      textarea.setSelectionRange(0, textarea.value.length);

      try {
        if (!document.execCommand("copy")) {
          throw new Error("Copy command failed");
        }
        resolve();
      } catch (error) {
        reject(error);
      } finally {
        textarea.remove();
      }
    });
  }

  function copyText(value) {
    if (navigator.clipboard && window.isSecureContext) {
      return navigator.clipboard.writeText(value).catch(function () {
        return fallbackCopy(value);
      });
    }
    return fallbackCopy(value);
  }

  function closeWallet() {
    if (!walletDialog || !walletDialog.open) {
      return;
    }
    if (typeof walletDialog.close === "function") {
      walletDialog.close();
    } else {
      walletDialog.removeAttribute("open");
      document.body.classList.remove("wallet-open");
    }
  }

  function openWallet() {
    if (!walletDialog || walletDialog.open) {
      return;
    }
    if (mapShell && mapShell.classList.contains("is-expanded")) {
      setMapExpanded(false);
    }
    lastWalletOpener = document.activeElement;
    document.body.classList.add("wallet-open");
    if (typeof walletDialog.showModal === "function") {
      walletDialog.showModal();
    } else {
      walletDialog.setAttribute("open", "");
      walletCloseButton.focus();
    }
  }

  function initializeWallet() {
    if (!walletDialog || !walletOpenButton) {
      return;
    }

    renderWallet();
    walletOpenButton.addEventListener("click", openWallet);
    walletCloseButton.addEventListener("click", closeWallet);
    walletDialog.addEventListener("click", function (event) {
      if (event.target === walletDialog) {
        closeWallet();
      }

      var copyButton = event.target.closest && event.target.closest(".wallet-copy");
      if (!copyButton || copyButton.disabled) {
        return;
      }

      var originalLabel = copyButton.textContent;
      copyText(copyButton.dataset.copyValue).then(function () {
        copyButton.textContent = "완료";
        walletStatus.textContent = copyButton.dataset.copyLabel + "을 클립보드에 복사했습니다.";
        window.setTimeout(function () { copyButton.textContent = originalLabel; }, 1600);
      }).catch(function () {
        walletStatus.textContent = "복사하지 못했습니다. 값을 길게 눌러 복사해 주세요.";
      });
    });
    walletDialog.addEventListener("close", function () {
      document.body.classList.remove("wallet-open");
      if (!document.body.classList.contains("trip-locked") && lastWalletOpener && typeof lastWalletOpener.focus === "function") {
        lastWalletOpener.focus();
      }
    });
  }

  function initializeTodayMode() {
    updateJourneyNow();
    liveTimer = window.setInterval(updateJourneyNow, 60 * 1000);
    document.addEventListener("visibilitychange", function () {
      if (document.visibilityState === "visible") {
        updateJourneyNow();
      }
    });
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
    initializeDirections();
    initializeWallet();
    initializeTodayMode();
    initializeReveal();
    initializeMap();
  }

  document.addEventListener("trip:unlocked", initializeTripApp);
  document.addEventListener("trip:locked", function () {
    closeWallet();
    if (mapShell && mapShell.classList.contains("is-expanded")) {
      setMapExpanded(false);
    }
  });

  if (!document.body.classList.contains("trip-locked")) {
    initializeTripApp();
  }
}());
