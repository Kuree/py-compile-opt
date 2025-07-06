#ifndef PY_COMPILE_OPT_INITALLTRANSLATIONS_HH
#define PY_COMPILE_OPT_INITALLTRANSLATIONS_HH

namespace mlir {
void registerFromPycTranslation();

inline void registerAllTranslations() {
    static bool initOnce = []() {
        registerFromPycTranslation();
        return true;
    }();
    (void)initOnce;
}
} // namespace mlir

#endif // PY_COMPILE_OPT_INITALLTRANSLATIONS_HH
