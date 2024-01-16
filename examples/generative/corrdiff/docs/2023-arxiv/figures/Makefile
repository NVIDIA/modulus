all: diffusion_fig_front.pdf typhoon_statistics.pdf reflectivity_maps.png spectra_and_distributions.pdf typhoon_koinu_radar.pdf autocorrelation time_means benchmark.txt scores.txt

diffusion_fig_front.pdf:
	@echo "This takes a few minutes"
	python plot_front.py

typhoon_statistics.pdf:
	@echo "This takes a few minutes"
	python plot_historical_typhoons.py

reflectivity_maps.png:
	python plot_reflectivity_maps.py

spectra_and_distributions.pdf:
	@echo "This takes a few minutes"
	python plot_spectra_logPDFs.py

# also saves typhoon_forecasts_axis.pdf
typhoon_koinu_radar.pdf:
	python plot_typhoon_forecast.py

autocorrelation:
	python autocorrelation.py

time_means:
	python3 time_means.py

benchmark.txt:
	python3 benchmark.py | tee $@

scores.txt:
	python3 scores_for_table.py | tee $@
