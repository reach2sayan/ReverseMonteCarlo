#pragma once
#include <boost/log/sinks/basic_sink_backend.hpp>
#include <boost/log/sinks/sync_frontend.hpp>
#include <boost/log/trivial.hpp>
#include <boost/shared_ptr.hpp>

#include <cstddef>
#include <deque>
#include <mutex>
#include <string>
#include <vector>

namespace rmcgui {

enum class LogSeverity { Trace, Debug, Info, Warning, Error, Fatal };

struct LogLine {
  LogSeverity sev;
  std::string text;
};

// Thread-safe bounded log buffer. The Boost.Log sink (worker/main threads)
// pushes; the LogPanel (render thread) snapshots under the same lock.
class LogRing {
public:
  void push(LogSeverity sev, std::string text);
  [[nodiscard]] std::vector<LogLine> snapshot() const;
  void clear();
  // Monotonic counter of total lines ever pushed — lets the panel detect growth
  // for auto-scroll without diffing contents.
  [[nodiscard]] std::size_t version() const;

private:
  mutable std::mutex mtx_;
  std::deque<LogLine> lines_;
  std::size_t version_ = 0;
  static constexpr std::size_t kCap = 5000;
};

// Boost.Log backend that appends each formatted record to a LogRing.
class GuiLogBackend : public boost::log::sinks::basic_formatted_sink_backend<
                          char, boost::log::sinks::synchronized_feeding> {
public:
  explicit GuiLogBackend(LogRing &ring) : ring_(ring) {}
  void consume(const boost::log::record_view &rec,
               const string_type &formatted);

private:
  LogRing &ring_;
};

using GuiLogSink = boost::log::sinks::synchronous_sink<GuiLogBackend>;

// Install a sink feeding `ring`; keep the returned handle to remove it later.
boost::shared_ptr<GuiLogSink> install_gui_log_sink(LogRing &ring);
// Flush + detach the sink. Call AFTER the worker thread is joined and BEFORE
// the LogRing it references is destroyed.
void remove_gui_log_sink(const boost::shared_ptr<GuiLogSink> &sink);

} // namespace rmcgui
