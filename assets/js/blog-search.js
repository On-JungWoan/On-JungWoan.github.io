(function () {
  'use strict';

  var input = document.getElementById('search');
  var results = document.getElementById('results');
  var status = document.getElementById('search-status');
  if (!input || !results || !status) return;
  if (typeof lunr === 'undefined' || typeof store === 'undefined') {
    status.textContent = '검색을 불러오지 못했습니다. 잠시 후 다시 시도해 주세요.';
    return;
  }

  var labels = {};
  var categoryData = document.getElementById('blog-category-data');
  if (categoryData) {
    JSON.parse(categoryData.textContent).forEach(function (group) {
      group.items.forEach(function (category) { labels[category.key] = category.label; });
    });
  }
  var topicLabels = {};
  var topicData = document.getElementById('blog-topic-data');
  if (topicData) {
    JSON.parse(topicData.textContent).forEach(function (topic) { topicLabels[topic.key] = topic.label; });
  }
  var index = lunr(function () {
    this.field('title');
    this.field('excerpt');
    this.field('categories');
    this.field('tags');
    this.field('series');
    this.ref('id');
    this.pipeline.remove(lunr.trimmer);
    store.forEach(function (entry, id) {
      this.add({
        title: entry.title,
        excerpt: entry.excerpt,
        categories: (entry.categories || []).map(function (key) { return key + ' ' + (labels[key] || ''); }).join(' '),
        tags: (entry.tags || []).map(function (key) { return key + ' ' + (topicLabels[key] || ''); }).join(' '),
        series: entry.series_label || '',
        id: id
      });
    }, this);
  });

  function element(tag, className, text) {
    var node = document.createElement(tag);
    node.className = className;
    if (text) node.textContent = text;
    return node;
  }

  function search() {
    var query = input.value.trim().toLowerCase();
    results.replaceChildren();
    var url = new URL(window.location.href);
    if (query) url.searchParams.set('q', input.value.trim());
    else url.searchParams.delete('q');
    window.history.replaceState(null, '', url);
    if (!query) {
      status.textContent = '검색어를 입력하면 관련 글을 보여드립니다.';
      return;
    }
    var matches = index.query(function (q) {
      query.split(lunr.tokenizer.separator).filter(Boolean).forEach(function (term) {
        q.term(term, { boost: 100 });
        q.term(term, { usePipeline: false, wildcard: lunr.Query.wildcard.TRAILING, boost: 10 });
        q.term(term, { usePipeline: false, editDistance: 1, boost: 1 });
      });
    });
    // Korean titles often join a prefix and a word, e.g. "[Jekyll]사이드바".
    // Show literal title matches first, followed by Lunr's ranked matches.
    var titleMatches = [];
    var titleRefs = new Set();
    store.forEach(function (entry, id) {
      if ((entry.title || '').toLowerCase().includes(query)) {
        titleMatches.push({ ref: String(id) });
        titleRefs.add(String(id));
      }
    });
    matches = titleMatches.concat(matches.filter(function (match) { return !titleRefs.has(String(match.ref)); }));
    status.textContent = matches.length ? matches.length + '개의 글을 찾았습니다.' : '검색 결과가 없습니다. 다른 키워드로 찾아보세요.';
    var fragment = document.createDocumentFragment();
    matches.forEach(function (match) {
      var entry = store[match.ref];
      var article = element('article', 'blog-post');
      var body = element('div', 'blog-post__body');
      var meta = element('div', 'blog-post__meta');
      var category = entry.categories && entry.categories[0];
      if (category) meta.appendChild(element('span', 'blog-post__category', labels[category] || category));
      if (entry.series_label && entry.series_url) {
        var series = element('a', 'blog-post__series', entry.series_label);
        series.href = entry.series_url;
        meta.appendChild(series);
      }
      if (entry.date) {
        var date = element('time', '', entry.date.replace(/-/g, '.'));
        date.dateTime = entry.date;
        meta.appendChild(date);
      }
      var heading = element('h2', 'blog-post__title');
      var link = element('a', '', entry.title);
      link.href = entry.url;
      heading.appendChild(link);
      body.appendChild(meta);
      body.appendChild(heading);
      body.appendChild(element('p', 'blog-post__excerpt', entry.summary || entry.excerpt));
      article.appendChild(body);
      fragment.appendChild(article);
    });
    results.appendChild(fragment);
  }

  input.addEventListener('input', search);
  input.value = new URLSearchParams(window.location.search).get('q') || '';
  search();
})();
