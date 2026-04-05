#pragma once

#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <sstream>
#include <string>

namespace edge {

enum class LogLevel { Debug, Info, Warn, Error };

inline const char* level_str(LogLevel l) {
  switch (l) {
    case LogLevel::Debug:
      return "DEBUG";
    case LogLevel::Info:
      return "INFO";
    case LogLevel::Warn:
      return "WARN";
    case LogLevel::Error:
      return "ERROR";
    default:
      return "INFO";
  }
}

class Logger {
 public:
  static Logger& instance() {
    static Logger g;
    return g;
  }

  void set_file(const std::string& path) {
    std::lock_guard<std::mutex> lk(mu_);
    if (ofs_.is_open()) {
      ofs_.close();
    }
    if (!path.empty()) {
      ofs_.open(path, std::ios::app);
    }
  }

  void set_min_level(LogLevel l) { min_ = l; }

  void log(LogLevel level, const std::string& component, const std::string& msg) {
    if (level < min_) {
      return;
    }
    const auto now = std::chrono::system_clock::now();
    const auto t = std::chrono::system_clock::to_time_t(now);
    std::tm tm_buf {};
#if defined(_WIN32)
    localtime_s(&tm_buf, &t);
#else
    localtime_r(&t, &tm_buf);
#endif
    std::ostringstream oss;
    oss << std::put_time(&tm_buf, "%Y-%m-%d %H:%M:%S") << " [" << level_str(level) << "] [" << component
        << "] " << msg;
    const auto line = oss.str();
    std::lock_guard<std::mutex> lk(mu_);
    std::cerr << line << std::endl;
    if (ofs_.is_open()) {
      ofs_ << line << std::endl;
    }
  }

 private:
  Logger() = default;
  std::mutex mu_;
  std::ofstream ofs_;
  LogLevel min_{LogLevel::Info};
};

inline void log_info(const std::string& c, const std::string& m) { Logger::instance().log(LogLevel::Info, c, m); }
inline void log_warn(const std::string& c, const std::string& m) { Logger::instance().log(LogLevel::Warn, c, m); }
inline void log_error(const std::string& c, const std::string& m) { Logger::instance().log(LogLevel::Error, c, m); }

}  // namespace edge
