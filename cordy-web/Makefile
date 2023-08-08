.DEFAULT_GOAL = main
.PHONY : main
main :
	wasm-pack build --release --target web
	cp pkg/cordy_web.js ../../../Website/public/cordy/pkg/.
	cp pkg/cordy_web_bg.wasm ../../../Website/public/cordy/pkg/.
