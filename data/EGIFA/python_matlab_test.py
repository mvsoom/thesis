import matlab.engine

eng = matlab.engine.start_matlab()
eng.cd("/home/marnix/pro/science/thesis/data/EGIFA", nargout=0)

out = eng.evalc("score_vtl_continuous1('speech/res_oracle/')")

print(out)

eng.quit()
