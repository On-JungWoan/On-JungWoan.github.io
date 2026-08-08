---
---
(function () {
  "use strict";

  var data = {{ site.data.japan_trip | jsonify }};

  data.liveEvents = data.days.reduce(function (events, day) {
    return events.concat((day.events || []).map(function (event) {
      event.day = day.id;
      return event;
    }));
  }, []);

  window.JAPAN_TRIP_DATA = data;
}());
