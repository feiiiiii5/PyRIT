import type { RegisteredInitializer } from '@/types'

export const UNREGISTERED_INITIALIZER_DESCRIPTION = 'Initializer is no longer registered.'
export const CATALOG_UNAVAILABLE_DESCRIPTION =
  'Initializer catalog is temporarily unavailable; registration state cannot be confirmed.'

/**
 * Resolve a settings entry's `initializer_name` to its catalog definition.
 *
 * Settings reference an initializer by name; the catalog (from `listRegistered`)
 * is the single source of truth for display metadata. When a persisted name is no
 * longer registered, return a placeholder so the row still renders.
 *
 * `catalogAvailable` distinguishes the two ways a name can fail to resolve:
 * a successful catalog response that lacks the name means the entry is truly
 * unregistered, while a failed catalog request only means registration state
 * is unknown and must not be reported as a configuration problem.
 */
export function resolveRegisteredInitializer(
  initializerName: string,
  registeredInitializers: RegisteredInitializer[],
  catalogAvailable: boolean = true,
): RegisteredInitializer {
  if (!catalogAvailable) {
    return {
      initializer_name: initializerName,
      initializer_type: 'UnverifiedInitializer',
      description: CATALOG_UNAVAILABLE_DESCRIPTION,
      required_env_vars: [],
      supported_parameters: [],
    }
  }

  const match = registeredInitializers.find((item) => item.initializer_name === initializerName)
  if (match) {
    return match
  }

  return {
    initializer_name: initializerName,
    initializer_type: 'UnknownInitializer',
    description: UNREGISTERED_INITIALIZER_DESCRIPTION,
    required_env_vars: [],
    supported_parameters: [],
  }
}
