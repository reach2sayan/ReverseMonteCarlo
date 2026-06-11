#include "LogSink.hpp"

#include <boost/log/core.hpp>
#include <boost/log/expressions.hpp>
#include <boost/make_shared.hpp>

#include <utility>

namespace rmcgui {

void LogRing::push(LogSeverity sev, std::string text) {
  std::scoped_lock lk(mtx_);
  lines_.push_back({sev, std::move(text)});
  if (lines_.size() > kCap) {
    lines_.pop_front();
  }
  ++version_;
}

std::vector<LogLine> LogRing::snapshot() const {
  std::scoped_lock lk(mtx_);
  return {lines_.begin(), lines_.end()};
}

void LogRing::clear() {
  std::scoped_lock lk(mtx_);
  lines_.clear();
  ++version_;
}

std::size_t LogRing::version() const {
  std::scoped_lock lk(mtx_);
  return version_;
}

namespace {
LogSeverity map_severity(boost::log::trivial::severity_level lvl) {
  switch (lvl) {
  case boost::log::trivial::trace:
    return LogSeverity::Trace;
  case boost::log::trivial::debug:
    return LogSeverity::Debug;
  case boost::log::trivial::info:
    return LogSeverity::Info;
  case boost::log::trivial::warning:
    return LogSeverity::Warning;
  case boost::log::trivial::error:
    return LogSeverity::Error;
  case boost::log::trivial::fatal:
    return LogSeverity::Fatal;
  }
  return LogSeverity::Info;
}
} // namespace

void GuiLogBackend::consume(const boost::log::record_view &rec,
                            const string_type &formatted) {
  LogSeverity sev = LogSeverity::Info;
  if (auto lvl = rec[boost::log::trivial::severity]) {
    sev = map_severity(*lvl);
  }
  ring_.push(sev, formatted);
}

boost::shared_ptr<GuiLogSink> install_gui_log_sink(LogRing &ring) {
  auto backend = boost::make_shared<GuiLogBackend>(ring);
  auto sink = boost::make_shared<GuiLogSink>(backend);
  // We only want the message text in the ring; severity is carried separately.
  sink->set_formatter(boost::log::expressions::stream
                      << boost::log::expressions::smessage);
  boost::log::core::get()->add_sink(sink);
  return sink;
}

void remove_gui_log_sink(const boost::shared_ptr<GuiLogSink> &sink) {
  if (!sink) {
    return;
  }
  auto core = boost::log::core::get();
  core->flush();
  core->remove_sink(sink);
}

} // namespace rmcgui
