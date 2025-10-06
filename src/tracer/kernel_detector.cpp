
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <sys/stat.h>
#include <unistd.h>

#include <cuda.h>
#include <cupti.h>

#include "spdlog/sinks/basic_file_sink.h"

#define ENV_KERNEL_LOGFILE "KERNEL_LOGFILE"

extern "C"
{
#define CUPTI_CALL(call)                                                         \
    {                                                                            \
        CUptiResult _status = call;                                              \
        if (_status != CUPTI_SUCCESS)                                            \
        {                                                                        \
            const char *errstr;                                                  \
            cuptiGetResultString(_status, &errstr);                              \
            fprintf(stderr, "%s:%d: error: function %s failed with error %s.\n", \
                    __FILE__, __LINE__, #call, errstr);                          \
            exit(-1);                                                            \
        }                                                                        \
    }
}

CUpti_SubscriberHandle subscriber;
char *kernel_log_path;
std::shared_ptr<spdlog::logger> logger;

void CUPTIAPI
callbackHandler(void *userdata, CUpti_CallbackDomain domain,
                CUpti_CallbackId cbid, void *cbdata)
{
    const CUpti_CallbackData *cbInfo = (CUpti_CallbackData *)cbdata;

    CUPTI_CALL(cuptiGetLastError());

    if (cbInfo->callbackSite == CUPTI_API_ENTER)
    {
        if (cbid == CUPTI_DRIVER_TRACE_CBID_cuModuleGetFunction)
        {
            cuModuleGetFunction_params_st *params = (cuModuleGetFunction_params_st *)(cbInfo->functionParams);
            const char *kernel_name = params->name;
            logger->info("{}", kernel_name);
        }
    }
}

static CUptiResult
cuptiInitialize(void)
{
    CUptiResult status = CUPTI_SUCCESS;

    CUPTI_CALL(cuptiSubscribe(&subscriber, (CUpti_CallbackFunc)callbackHandler, NULL));

    CUPTI_CALL(cuptiEnableDomain(1, subscriber, CUPTI_CB_DOMAIN_DRIVER_API));
    CUPTI_CALL(cuptiEnableDomain(1, subscriber, CUPTI_CB_DOMAIN_RUNTIME_API));

    return status;
}

extern "C" int InitializeInjection(void)
{
    subscriber = 0;
    kernel_log_path = std::getenv(ENV_KERNEL_LOGFILE);

    if (kernel_log_path == NULL)
    {
        std::cerr << "Environment variable " << ENV_KERNEL_LOGFILE << " not set." << std::endl;
        exit(1);
    }

    struct stat buffer;
    int exist = stat(kernel_log_path, &buffer);
    if (exist != 0)
    {
        std::cerr << "Log file path " << kernel_log_path << " does not exist." << std::endl;
        exit(1);
    }

    try
    {
        logger = spdlog::basic_logger_mt("kernel_logger", kernel_log_path);
        logger->set_pattern("%v");
    }
    catch (const spdlog::spdlog_ex &ex)
    {
        std::cerr << "Log init failed: " << ex.what() << std::endl;
        exit(1);
    }

    CUPTI_CALL(cuptiInitialize());
    std::cout << "CUPTI libraray injected" << std::endl;

    return 0;
}
