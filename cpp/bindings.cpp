#include <pybind11/functional.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "engine.hpp"

namespace py = pybind11;
using namespace qbt;

using Arr = py::array_t<double, py::array::c_style | py::array::forcecast>;

static std::vector<double> to_vec(const Arr& a, py::ssize_t T, py::ssize_t N, const char* name) {
  if (a.ndim() != 2 || a.shape(0) != T || a.shape(1) != N)
    throw std::invalid_argument(std::string(name) + " must have shape (T, N)");
  return {a.data(), a.data() + a.size()};
}

static py::array_t<double> to_arr(const std::vector<double>& v, std::vector<py::ssize_t> shape) {
  py::array_t<double> out(shape);
  std::copy(v.begin(), v.end(), out.mutable_data());
  return out;
}

PYBIND11_MODULE(_core, m) {
  m.doc() = "C++ event-driven execution core for qbt";

  py::class_<Config>(m, "Config")
      .def(py::init<>())
      .def(py::init([](py::kwargs kw) {
        Config c;
        py::object o = py::cast(&c, py::return_value_policy::reference);
        for (auto [k, v] : kw) o.attr(k) = v;
        return c;
      }))
      .def_readwrite("initial_cash", &Config::initial_cash)
      .def_readwrite("commission_bps", &Config::commission_bps)
      .def_readwrite("half_spread_bps", &Config::half_spread_bps)
      .def_readwrite("impact_coef", &Config::impact_coef)
      .def_readwrite("max_participation", &Config::max_participation)
      .def_readwrite("latency_bars", &Config::latency_bars)
      .def_readwrite("min_trade_value", &Config::min_trade_value);

  py::class_<Engine>(m, "Engine")
      .def(py::init([](Arr open, Arr high, Arr low, Arr close, Arr volume, Config cfg) {
             if (close.ndim() != 2) throw std::invalid_argument("close must be 2-D (T, N)");
             auto T = close.shape(0), N = close.shape(1);
             return Engine(static_cast<int>(T), static_cast<int>(N), to_vec(open, T, N, "open"),
                           to_vec(high, T, N, "high"), to_vec(low, T, N, "low"), to_vec(close, T, N, "close"),
                           to_vec(volume, T, N, "volume"), cfg);
           }),
           py::arg("open"), py::arg("high"), py::arg("low"), py::arg("close"), py::arg("volume"),
           py::arg("config") = Config{})
      // strategy-facing API
      .def("submit", &Engine::submit, py::arg("sym"), py::arg("qty"), py::arg("limit") = py::none(),
           py::arg("tif_bars") = 0, "Submit a market (no limit) or limit order; returns order id")
      .def("cancel", &Engine::cancel)
      .def("cancel_all", &Engine::cancel_all)
      .def("position", &Engine::position)
      .def("pending_qty", &Engine::pending_qty)
      .def_property_readonly("positions", [](const Engine& e) { return to_arr(e.positions(), {e.N()}); })
      .def_property_readonly("cash", &Engine::cash)
      .def_property_readonly("equity", &Engine::equity)
      .def_property_readonly("t", &Engine::now)
      // drivers
      .def("run", [](Engine& e, const Engine::Strategy& s) { e.run(s); }, py::arg("on_bar"),
           "Event-driven run; on_bar(engine, t) is called at each bar close")
      .def("run_targets", [](Engine& e, Arr w) {
             auto v = to_vec(w, e.T(), e.N(), "weights");
             py::gil_scoped_release release;  // pure C++ loop
             e.run_targets(v);
           }, py::arg("weights"), "Rebalance to target weights (T, N) at each close")
      // results
      .def("results", [](const Engine& e) {
        const auto& f = e.fills();
        py::array_t<int64_t> ft(f.size()), fs(f.size()), fid(f.size());
        py::array_t<double> fq(f.size()), fp(f.size()), fc(f.size());
        for (std::size_t i = 0; i < f.size(); ++i) {
          ft.mutable_at(i) = f[i].t; fs.mutable_at(i) = f[i].sym; fid.mutable_at(i) = f[i].order_id;
          fq.mutable_at(i) = f[i].qty; fp.mutable_at(i) = f[i].price; fc.mutable_at(i) = f[i].commission;
        }
        py::dict fills;
        fills["t"] = ft; fills["sym"] = fs; fills["order_id"] = fid;
        fills["qty"] = fq; fills["price"] = fp; fills["commission"] = fc;
        py::dict out;
        out["equity"] = to_arr(e.equity_curve(), {e.T()});
        out["cash"] = to_arr(e.cash_curve(), {e.T()});
        out["positions"] = to_arr(e.position_history(), {e.T(), e.N()});
        out["fills"] = fills;
        out["events"] = e.events_processed();
        return out;
      });
}
