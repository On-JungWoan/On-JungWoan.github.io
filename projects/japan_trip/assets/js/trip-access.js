(function () {
  "use strict";

  var STORAGE_KEY = "japan-trip-access-v1";
  var STORAGE_VALUE = "granted";
  var PASSWORD_HASH = "b3c976e29994fa2e7ed556c50c26c344ed0d6ebc198c439fde99280d00a84205";
  var DEFAULT_MESSAGE = "6자리 비밀번호를 입력하세요.";
  var body = document.body;
  var gate = document.getElementById("access-gate");
  var form = document.getElementById("access-form");
  var input = document.getElementById("trip-password");
  var toggle = document.querySelector(".password-toggle");
  var submit = document.querySelector(".access-submit");
  var message = document.getElementById("access-message");
  var privateContent = Array.from(document.querySelectorAll("[data-private-content]"));
  var relockButtons = Array.from(document.querySelectorAll(".relock-button"));
  var unlockTimer;

  if (!gate || !form || !input || !toggle || !submit || !message) {
    return;
  }

  function setPrivateLocked(locked) {
    privateContent.forEach(function (element) {
      element.inert = locked;
      if (locked) {
        element.setAttribute("aria-hidden", "true");
      } else {
        element.removeAttribute("aria-hidden");
      }
    });
  }

  function setMessage(text, state) {
    form.classList.toggle("has-error", state === "error");
    form.classList.toggle("is-success", state === "success");
    input.setAttribute("aria-invalid", state === "error" ? "true" : "false");
    message.textContent = text;
  }

  function hasStoredAccess() {
    try {
      return window.sessionStorage.getItem(STORAGE_KEY) === STORAGE_VALUE;
    } catch (error) {
      return false;
    }
  }

  function storeAccess() {
    try {
      window.sessionStorage.setItem(STORAGE_KEY, STORAGE_VALUE);
    } catch (error) {
      // Access still works for this page view when storage is unavailable.
    }
  }

  function clearStoredAccess() {
    try {
      window.sessionStorage.removeItem(STORAGE_KEY);
    } catch (error) {
      // The current page can still be locked when storage is unavailable.
    }
  }

  function digest(value) {
    var bytes = new TextEncoder().encode(value);
    return window.crypto.subtle.digest("SHA-256", bytes).then(function (buffer) {
      return Array.from(new Uint8Array(buffer)).map(function (byte) {
        return byte.toString(16).padStart(2, "0");
      }).join("");
    });
  }

  function finishUnlock() {
    body.classList.remove("trip-locked");
    gate.hidden = true;
    gate.classList.remove("is-unlocking");
    setPrivateLocked(false);
    submit.disabled = false;
    document.dispatchEvent(new CustomEvent("trip:unlocked"));
  }

  function unlock(animate) {
    window.clearTimeout(unlockTimer);
    setMessage("확인되었습니다. 여행 일정을 엽니다.", "success");

    if (!animate) {
      finishUnlock();
      return;
    }

    gate.classList.add("is-unlocking");
    unlockTimer = window.setTimeout(finishUnlock, 360);
  }

  function lock() {
    window.clearTimeout(unlockTimer);
    clearStoredAccess();
    gate.hidden = false;
    gate.classList.remove("is-unlocking");
    body.classList.add("trip-locked");
    setPrivateLocked(true);
    input.value = "";
    input.type = "password";
    toggle.textContent = "보기";
    toggle.setAttribute("aria-pressed", "false");
    toggle.setAttribute("aria-label", "비밀번호 표시");
    setMessage(DEFAULT_MESSAGE);
    document.dispatchEvent(new CustomEvent("trip:locked"));
    window.requestAnimationFrame(function () { input.focus(); });
  }

  form.addEventListener("submit", function (event) {
    event.preventDefault();
    var password = input.value.trim();

    if (!/^\d{6}$/.test(password)) {
      setMessage("숫자 6자리를 입력해 주세요.", "error");
      input.focus();
      return;
    }

    submit.disabled = true;
    setMessage("비밀번호를 확인하고 있습니다.");

    digest(password).then(function (hash) {
      if (hash !== PASSWORD_HASH) {
        submit.disabled = false;
        setMessage("비밀번호가 올바르지 않습니다. 다시 확인해 주세요.", "error");
        input.focus();
        input.select();
        return;
      }

      storeAccess();
      unlock(true);
    }).catch(function () {
      submit.disabled = false;
      setMessage("이 브라우저에서는 비밀번호를 확인할 수 없습니다.", "error");
      input.focus();
    });
  });

  input.addEventListener("input", function () {
    input.value = input.value.replace(/\D/g, "").slice(0, 6);
    if (form.classList.contains("has-error")) {
      setMessage(DEFAULT_MESSAGE);
    }
  });

  toggle.addEventListener("click", function () {
    var willShow = input.type === "password";
    input.type = willShow ? "text" : "password";
    toggle.textContent = willShow ? "숨기기" : "보기";
    toggle.setAttribute("aria-pressed", String(willShow));
    toggle.setAttribute("aria-label", willShow ? "비밀번호 숨기기" : "비밀번호 표시");
    input.focus();
  });

  relockButtons.forEach(function (button) {
    button.addEventListener("click", lock);
  });

  if (hasStoredAccess()) {
    unlock(false);
  } else {
    gate.hidden = false;
    setPrivateLocked(true);
    window.requestAnimationFrame(function () { input.focus(); });
  }
}());
